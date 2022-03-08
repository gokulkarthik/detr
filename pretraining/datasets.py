import os
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from dall_e import map_pixels, load_model


class ImagenetForSSL(ImageFolder):
    """
    Note: Common placeholder label is used for all the validation set images. Don't use the imagenet label of validation set.

    Refs:
    patch: https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/11
    """
    def __init__(self, img_folder, feature_extractor, image_size, patch_size, pretext_task, pretext_task_ratio):
        super(ImagenetForSSL, self).__init__(img_folder)
        self.feature_extractor = feature_extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.pretext_task = pretext_task
        self.pretext_task_ratio = pretext_task_ratio
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.transforms = T.Compose([T.Resize(int(self.image_size*1.2)), 
            T.RandomCrop(self.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)])
        self.transforms_for_dalle = T.Compose([T.Resize(int(self.image_size*(1.2/4))), 
            T.RandomCrop(self.image_size//4),
            T.RandomHorizontalFlip(), 
            T.ToTensor()])
        if self.pretext_task == 'mim-discrete':
            self.dalle_encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", torch.device('cpu'))

    def __getitem__(self, idx):
        img, label = super(ImagenetForSSL, self).__getitem__(idx)
        
        #encoding = self.feature_extractor(images=img, return_tensors="pt") # resizing to [H, W] and normalization with Imagenet mean
        #pixel_values = encoding["pixel_values"] # [1, 3, H, W]
        
        pixel_values = self.transforms(img).unsqueeze(0) # [1, 3, H, W]

        _, _, H, W = pixel_values.shape
        patches = pixel_values.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # [1, 3, h, w, patch_size_h, patch_size_w]
        unfold_shape = patches.size()
        patches = patches.contiguous().view(1, 3, -1, self.patch_size, self.patch_size)  # [1, 3, num_patches=hw, patch_size_h, patch_size_w]
        num_patches = patches.shape[2]

        if self.pretext_task == 'mim-continuous' or  self.pretext_task == 'jigsaw-continuous':
            patches_for_reconstruction = patches.permute(0, 2, 3, 4, 1).squeeze() # [num_patches=hw, patch_size_h, patch_size_w, 3]
            patches_for_reconstruction = patches_for_reconstruction.reshape(num_patches, -1) # [num_patches=hw, patch_size_h, patch_size_w, 3] -> # [num_patches=hw, patch_size_h*patch_size_w*3]

        selected_patches = np.random.choice(num_patches, size=int(self.pretext_task_ratio*num_patches), replace=False)

        # format input
        if self.pretext_task.startswith('mim'):
            patches[:, 0, selected_patches, :, :] = self.imagenet_mean[0]
            patches[:, 1, selected_patches, :, :] = self.imagenet_mean[1]
            patches[:, 2, selected_patches, :, :] = self.imagenet_mean[2]
            patches_info = torch.zeros(num_patches).long()
            patches_info[selected_patches] = 1 # shuffled or not
        elif self.pretext_task.startswith('jigsaw'):
            selected_patches_sorted = sorted(selected_patches)
            patches[:, :, selected_patches_sorted, :, :] =  patches[:, :, selected_patches, :, :]
            patches_info = torch.arange(num_patches) 
            patches_info[selected_patches_sorted] = torch.Tensor(selected_patches).long() # original position

        # format output
        if self.pretext_task == 'mim-discrete':
            pixel_values_for_dalle = map_pixels(self.transforms_for_dalle(img).unsqueeze(0))
            z_logits = self.dalle_encoder(pixel_values_for_dalle)
            target = torch.argmax(z_logits, axis=1).flatten()
        elif self.pretext_task == 'jigsaw-discrete':
            target = patches_info
        elif self.pretext_task == 'mim-continuous':
            target = patches_for_reconstruction
        elif self.pretext_task == 'jigsaw-continuous':
            target = patches_for_reconstruction
        else:
            raise ValueError()

        # patches from [1, 3, num_patches=hw, patch_size_h, patch_size_w] to [1, 3, h, w, patch_size_h, patch_size_w]
        patches = patches.view(unfold_shape)
        # patches from [1, 3, h, w, patch_size_h, patch_size_w] to [1, 3, h, patch_size_h, w, patch_size_w]
        patches = patches.permute(0, 1, 2, 4, 3, 5)
        # patches from [1, 3, h, patch_size_h, w, patch_size_w] to [1, 3, H, W]
        patches = patches.reshape(1, 3, H, W)

        item = {
            #'pixel_values': pixel_values[0], # [3, H, W] # original
            'patches': patches[0], # [3, H, W] # transformed
            'patches_info': patches_info, # [num_patches] # target for jigsaw-discrete
            'target': target, # [num_patches] for {jigsaw-discrete, mim-discrete} with each value denoting the class; [num_patches, patch_size_h*patch_size_w*3] for {jigsaw-continuous, mim-continuous}
        }

        return item

def load_dataset(args, split, feature_extractor):

    if args.dataset == 'imagenet':
        if split == 'val':
            split = 'val-with-subfolder'
        dataset = ImagenetForSSL(img_folder=f'../data/imagenet/ILSVRC/Data/CLS-LOC/{split}', 
            feature_extractor=feature_extractor, 
            image_size=args.max_image_size, 
            patch_size=args.patch_size, 
            pretext_task=args.pretext_task, 
            pretext_task_ratio=args.pretext_task_ratio)
    else:
        raise NotImplementedError

    return dataset