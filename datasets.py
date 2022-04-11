import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from dall_e import map_pixels, load_model


class IsaidDetectionOld(torchvision.datasets.CocoDetection):
    def __init__(self, root_folder, feature_extractor, split='train'):
        img_folder = os.path.join(root_folder, split, 'images')
        ann_file = os.path.join(root_folder, split, f'instancesonly_filtered_{split}.json')
        super(IsaidDetectionOld, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(IsaidDetectionOld, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

class IsaidDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root_folder, feature_extractor, split, args):
        img_folder = os.path.join(root_folder, split, 'images')
        ann_file = os.path.join(root_folder, split, f'instancesonly_filtered_{split}.json')
        super(IsaidDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

        self.args = args
        if args.ssl_task != "none":
            assert args.min_image_size == args.max_image_size
            self.image_size = args.min_image_size
            self.patch_size = args.ssl_patch_size
            self.ssl_task = args.ssl_task
            self.ssl_task_ratio = args.ssl_task_ratio
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]
            self.ssl_transforms = T.Compose([T.Resize((self.image_size, self.image_size)),
                #T.Resize(int(self.image_size*1.2)), 
                #T.RandomCrop(self.image_size),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)])
            self.ssl_transforms_for_dalle = T.Compose([T.Resize((self.image_size, self.image_size)),
                #[T.Resize(int(self.image_size*(1.2/4))), 
                #T.RandomCrop(self.image_size//4),
                #T.RandomHorizontalFlip(), 
                T.ToTensor()])
            if self.ssl_task == 'mim-discrete':
                self.ssl_dalle_encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", torch.device('cpu'))

    def get_item_for_ssl(self, img):
        pixel_values = self.ssl_transforms(img).unsqueeze(0) # [1, 3, H, W]
        _, _, H, W = pixel_values.shape
        patches = pixel_values.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # [1, 3, h, w, patch_size_h, patch_size_w]
        unfold_shape = patches.size()
        patches = patches.contiguous().view(1, 3, -1, self.patch_size, self.patch_size)  # [1, 3, num_patches=hw, patch_size_h, patch_size_w]
        num_patches = patches.shape[2]

        if self.ssl_task == 'mim-continuous' or  self.ssl_task == 'jigsaw-continuous':
            patches_for_reconstruction = patches.permute(0, 2, 3, 4, 1).squeeze() # [num_patches=hw, patch_size_h, patch_size_w, 3]
            patches_for_reconstruction = patches_for_reconstruction.reshape(num_patches, -1) # [num_patches=hw, patch_size_h, patch_size_w, 3] -> # [num_patches=hw, patch_size_h*patch_size_w*3]

        selected_patches = np.random.choice(num_patches, size=int(self.ssl_task_ratio*num_patches), replace=False)

        # format input
        if self.ssl_task.startswith('mim'):
            patches[:, 0, selected_patches, :, :] = self.imagenet_mean[0]
            patches[:, 1, selected_patches, :, :] = self.imagenet_mean[1]
            patches[:, 2, selected_patches, :, :] = self.imagenet_mean[2]
            patches_info = torch.zeros(num_patches).long()
            patches_info[selected_patches] = 1 # shuffled or not
        elif self.ssl_task.startswith('jigsaw'):
            selected_patches_sorted = sorted(selected_patches)
            patches[:, :, selected_patches_sorted, :, :] =  patches[:, :, selected_patches, :, :]
            patches_info = torch.arange(num_patches) 
            patches_info[selected_patches_sorted] = torch.Tensor(selected_patches).long() # original position

        # format output
        if self.ssl_task == 'mim-discrete':
            pixel_values_for_dalle = map_pixels(self.ssl_transforms_for_dalle(img).unsqueeze(0))
            z_logits = self.ssl_dalle_encoder(pixel_values_for_dalle)
            target = torch.argmax(z_logits, axis=1).flatten()
            patches_mask = (patches_info == 1).long()
        elif self.ssl_task == 'jigsaw-discrete':
            target = patches_info
            patches_mask = (patches_info == torch.arange(patches_info.shape[0], device=patches_info.device).long()).long()
        elif self.ssl_task == 'mim-continuous':
            target = patches_for_reconstruction
            patches_mask = (patches_info == 1).long()
        elif self.ssl_task == 'jigsaw-continuous':
            target = patches_for_reconstruction
            patches_mask = (patches_info == torch.arange(patches_info.shape[0], device=patches_info.device).long()).long()
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
            'patches_mask': patches_mask, # [num_patches] # each value is from {0, 1} denoting transformed or not
            'target': target, # [num_patches] for {jigsaw-discrete, mim-discrete} with each value denoting the class; [num_patches, patch_size_h*patch_size_w*3] for {jigsaw-continuous, mim-continuous}
        }

        return item['patches'], item['patches_mask'], item['target']

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(IsaidDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension # [1, 3, H, W] -> [3, H, W]
        target = encoding["labels"][0] # remove batch dimension

        # ssl output format
        if self.args.ssl_task != "none":
            ssl_patches, ssl_patches_mask, ssl_target = self.get_item_for_ssl(img)
        else:
            ssl_patches, ssl_patches_mask, ssl_target = None, None, None

        return pixel_values, target, ssl_patches, ssl_patches_mask, ssl_target

class CustomCollator(object):

    def __init__(self, feature_extractor, args):
        self.feature_extractor = feature_extractor
        self.args = args

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]

        if self.args.ssl_task != "none":
            ssl_patches = [item[2] for item in batch]
            ssl_patches_mask = torch.stack([item[3] for item in batch])
            ssl_target = torch.stack([item[4] for item in batch])

        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels

        if self.args.ssl_task != "none":
            ssl_encoding = self.feature_extractor.pad_and_create_pixel_mask(ssl_patches, return_tensors="pt")
            batch['ssl_patches'] = ssl_encoding['pixel_values'] 
            batch['ssl_patches_mask'] = ssl_patches_mask
            batch['ssl_target'] = ssl_target

        return batch


def load_dataset(args, split, feature_extractor):

    if args.dataset == 'isaid':
        dataset = IsaidDetection(root_folder='data/iSAID_patches', feature_extractor=feature_extractor, split=split, args=args)
    else:
        raise NotImplementedError

    return dataset

