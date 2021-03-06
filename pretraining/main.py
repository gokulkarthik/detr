import argparse
import os

import wandb

from datasets import load_dataset
from engine import create_model

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

    parser = argparse.ArgumentParser(description="Pre-Training and evaluation script for DETR's encoder")

    # dataset parameters
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'isaid'])
    parser.add_argument('--min_image_size', default=512, type=int)
    parser.add_argument('--max_image_size', default=512, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--pretext_task', type=str, default='jigsaw-discrete')
    parser.add_argument('--pretext_task_ratio', type=float, default=0.5)

    # model parameters
    parser.add_argument('--backbone_pretrained', type=str2bool, default=True)
    parser.add_argument('--backbone_freeze', type=str2bool, default=True)

    # training parameters
    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--loss_only_for_transformed', type=str2bool, default=True)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--val_check_interval', default=0.25)
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

    # load dataloader
    num_cpus = min(args.batch_size, 4) #(multiprocessing.cpu_count() // len(args.gpus))-1
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus)
    
    # create model
    seed_everything(42, workers=True)
    model = create_model(args)

    # # sanity check
    # batch = next(iter(dataloader_train))
    # outputs = model.common_step(batch)
    # print(outputs.shape)
    # quit()

    wandb_logger = WandbLogger(project="detr-pretraining", config=args)
    
    weights_save_path = os.path.join(f'checkpoints/{wandb_logger.experiment.name}')
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath=weights_save_path)
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