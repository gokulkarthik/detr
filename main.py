import argparse
import os

import wandb

from datasets import load_dataset, CustomCollator
from engine import create_model
from utils import visualize_example

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader

def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arg_parser():

    parser = argparse.ArgumentParser(description='Traning and evaluation script for object detection using DETR')

    # dataset parameters
    parser.add_argument('--dataset', default='isaid', choices=['isaid', 'coco'])
    parser.add_argument('--min_image_size', default=800, type=int)
    parser.add_argument('--max_image_size', default=800, type=int)

    # model parameters
    parser.add_argument('--num_labels', default=15, type=int)
    parser.add_argument('--mscoco_pretrained', type=str2bool, default=False)
    parser.add_argument('--freeze_backbone', type=str2bool, default=False)
    parser.add_argument('--freeze_encoder', type=str2bool, default=False)
    parser.add_argument('--encoder_init_ckpt', type=str, default='none', help='Encoder checkpoint path')

    # ssl task parameters
    parser.add_argument('--ssl_patch_size', default=32, type=int)
    parser.add_argument('--ssl_task', type=str, default='jigsaw-discrete')
    parser.add_argument('--ssl_task_ratio', type=float, default=0.5)
    parser.add_argument('--ssl_loss_only_for_transformed', type=str2bool, default=True)
    parser.add_argument('--ssl_loss_weight', type=float, default=1)

    # training parameters
    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

    # other parameters

    return parser

def main(args):
    
    # load dataset
    feature_extractor = DetrFeatureExtractor(size=args.min_image_size, max_size=args.max_image_size)
    dataset_train = load_dataset(args=args, split='train', feature_extractor=feature_extractor)
    dataset_val = load_dataset(args=args, split='val', feature_extractor=feature_extractor)
    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))
    print("Image shape:", dataset_train[0][0].shape)
    
    # # sanity check
    # image_folder = os.path.join('data/iSAID_patches', 'train', 'images')
    # random_id = np.random.randint(0, len(dataset_train))
    # visualize_example(dataset_train, image_folder, random_id)

    # load dataloader
    collate_fn = CustomCollator(feature_extractor, args)
    num_cpus = min(args.batch_size, 16) #(multiprocessing.cpu_count() // len(args.gpus))-1
    dataloader_train = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus)
    dataloader_val = DataLoader(dataset_val, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=num_cpus)
    
    # create model
    seed_everything(42, workers=True)
    model = create_model(args, dataset_val.coco, feature_extractor)

    # # sanity check
    # batch = next(iter(dataloader_train))
    # outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    # print(outputs.logits.shape)

    wandb_logger = WandbLogger(project="detr", config=args)
    weights_save_path = os.path.join(f'checkpoints/{wandb_logger.experiment.name}')
    checkpoint_callback = ModelCheckpoint(monitor="validation/loss")
    trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val, 
        logger=wandb_logger, log_every_n_steps=args.log_every_n_steps, val_check_interval=args.val_check_interval,
        strategy=args.strategy, weights_save_path=weights_save_path, callbacks=[checkpoint_callback],
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches, precision=args.precision,
        deterministic=False)
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == '__main__':

    parser = get_arg_parser()
    args = parser.parse_args()
    args.gpus = [int(id_) for id_ in args.gpus.split()]
    if args.strategy == 'ddp':
        args.strategy = DDPPlugin(find_unused_parameters=False)
    elif args.strategy == 'none':
        args.strategy = None

    main(args)