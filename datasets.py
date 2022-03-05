import os

import torchvision
from torch.utils.data import DataLoader


class IsaidDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root_folder, feature_extractor, split='train'):
        img_folder = os.path.join(root_folder, split, 'images')
        ann_file = os.path.join(root_folder, split, f'instancesonly_filtered_{split}.json')
        super(IsaidDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(IsaidDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


class CustomCollator(object):

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch


def load_dataset(args, split, feature_extractor):

    if args.dataset == 'isaid':
        dataset = IsaidDetection(root_folder='data/iSAID_patches', feature_extractor=feature_extractor, split=split)
    else:
        raise NotImplementedError

    return dataset

