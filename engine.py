import copy
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_only
from transformers import DetrConfig, DetrForObjectDetection
from torchmetrics import Accuracy
from torchvision import transforms
from unittest import result
from utils import CocoEvaluator


class Detr(pl.LightningModule):

  def __init__(self, args, dataset_val_coco, feature_extractor):
    super().__init__()
                                            
    if args.mscoco_pretrained:
      self.model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", 
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
        )
    else:
      self.source_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", 
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
        )
      self.model = DetrForObjectDetection(DetrConfig())
      self.model.config.num_labels = args.num_labels
      self.model.class_labels_classifier = copy.deepcopy(self.source_model.class_labels_classifier)
      self.model.model.backbone.load_state_dict(self.source_model.model.backbone.state_dict())
      self.source_model = None
      
    if args.encoder_init_ckpt != 'none':
      encoder_state_dict = torch.load(args.encoder_init_ckpt)['state_dict']

      encoder_state_dict_backbone = {'.'.join(k.split('.')[2:]):v for k,v in encoder_state_dict.items() if k.split('.')[1] == 'backbone'}
      encoder_state_dict_input_projection= {'.'.join(k.split('.')[2:]):v for k,v in encoder_state_dict.items() if k.split('.')[1] == 'input_projection'}
      encoder_state_dict_encoder = {'.'.join(k.split('.')[2:]):v for k,v in encoder_state_dict.items() if k.split('.')[1] == 'encoder'}

      self.model.model.backbone.load_state_dict(encoder_state_dict_backbone)
      self.model.model.input_projection.load_state_dict(encoder_state_dict_input_projection)
      self.model.model.encoder.load_state_dict(encoder_state_dict_encoder)
      
      print(f'Weights loaded from {args.encoder_init_ckpt} for backbone, input_projection and encoder')

    if args.freeze_backbone:
      for n, p in self.model.named_parameters():
        if "backbone" in n:
          p.requires_grad_(False)

    if args.freeze_encoder:
      for n, p in self.model.named_parameters():
        if "encoder" in n:
          p.requires_grad_(False)

    self.min_image_size = args.min_image_size
    self.max_image_size = args.max_image_size

    self.lr = args.lr
    self.lr_backbone = args.lr_backbone
    self.weight_decay = args.weight_decay

    self.dataset_val_coco = dataset_val_coco
    self.coco_evaluator = CocoEvaluator(self.dataset_val_coco, ['bbox'])
    self.feature_extractor = feature_extractor

    self.ssl_patch_size = args.ssl_patch_size
    self.ssl_task = args.ssl_task
    self.ssl_task_ratio = args.ssl_task_ratio
    self.ssl_loss_only_for_transformed = args.ssl_loss_only_for_transformed
    self.ssl_loss_weight = args.ssl_loss_weight
    if self.ssl_task == 'none':
      self.return_dict = False
    else:
      self.return_dict = True

    if args.ssl_task == 'jigsaw-discrete':
      self.ssl_pre_prediction_head = nn.Sequential(
        nn.Linear(self.model.config.d_model, self.model.config.d_model),
        nn.ReLU(),
      )
      assert self.min_image_size == self.max_image_size
      self.ssl_prediction_head = nn.Linear(self.model.config.d_model, (self.max_image_size//32)**2)
      self.ssl_criterion = nn.CrossEntropyLoss(reduction='none')
      self.ssl_metric = Accuracy()
    elif args.ssl_task == 'mim-discrete':
      self.ssl_pre_prediction_head = nn.Sequential(
        nn.Linear(self.model.config.d_model, self.model.config.d_model),
        nn.ReLU(),
      )
      self.ssl_prediction_head = nn.Linear(self.model.config.d_model, 8192)
      self.ssl_criterion = nn.CrossEntropyLoss(reduction='none')
      self.ssl_metric = Accuracy()
    elif args.ssl_task == 'mim-continuous' or args.ssl_task == 'jigsaw-continuous':
      #self.ssl_prediction_head = nn.Linear(self.model.config.d_model, 32*32*3)
      self.ssl_pre_prediction_head = nn.Sequential(
        nn.Linear(self.model.config.d_model, self.model.config.d_model),
        nn.ReLU(),
      )
      self.ssl_prediction_head = nn.ConvTranspose2d(
        self.model.config.d_model, 
        3, 
        kernel_size=(self.ssl_patch_size, self.ssl_patch_size), 
        stride=(self.ssl_patch_size, self.ssl_patch_size)
        )
      self.ssl_criterion = nn.L1Loss(reduction='none')
    
    self.inv_trans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
      transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
      ])

  def forward(self, pixel_values, pixel_mask):
    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    return outputs
     
  def common_step(self, batch, batch_idx):
    pixel_values = batch["pixel_values"]
    pixel_mask = batch["pixel_mask"]
    labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, return_dict=True)
    loss, loss_dict = outputs.loss, outputs.loss_dict

    if self.ssl_task != 'none':
      ssl_detr_outputs = self.model(pixel_values=batch['ssl_patches'], pixel_mask=pixel_mask, labels=labels, return_dict=True)
      encoder_output = ssl_detr_outputs.encoder_last_hidden_state # [batch_size, num_patches, hidden_dim]

      if self.ssl_task == 'jigsaw-discrete' or self.ssl_task == 'mim-discrete':
        ssl_output = self.ssl_pre_prediction_head(encoder_output) # [batch_size, num_patches, hidden_dim]
        ssl_output = self.ssl_prediction_head(ssl_output) # [batch_size, num_patches, num_classes]
        ssl_output = ssl_output.permute(0, 2, 1) # [batch_size, num_patches, num_classes] -> [batch_size, num_classes, num_patches]
        ssl_loss = self.ssl_criterion(ssl_output, batch['ssl_target'])
        ssl_pred = torch.argmax(ssl_output, dim=1) # [batch_size, num_patches]

      elif self.ssl_task == 'jigsaw-continuous' or self.ssl_task == 'mim-continuous':
        #ssl_output = self.ssl_prediction_head(encoder_output) # [batch_size, num_patches, num_pixel_locations]
        ssl_output = self.ssl_pre_prediction_head(encoder_output) # [batch_size, num_patches, hidden_dim]
        ssl_output = ssl_output.transpose(1, 2) # [batch_size, hidden_dim, num_patches]
        num_patch_rows = int(math.sqrt(ssl_output.size()[2]))
        ssl_output = ssl_output.unflatten(2, (num_patch_rows, num_patch_rows)) # [batch_size, hidden_dim, num_patch_rows(or)h, num_patch_cols(or)w]
        ssl_output = self.ssl_prediction_head(ssl_output) # [batch_size, 3, H, W] 
       
        batch_size, C, H, W = ssl_output.shape
        ssl_output = ssl_output.unfold(2, self.ssl_patch_size, self.ssl_patch_size).unfold(3, self.ssl_patch_size, self.ssl_patch_size) # [batch_size, 3, h, w, patch_size_h, patch_size_w]
        ssl_output = ssl_output.contiguous().view(batch_size, C, -1, self.ssl_patch_size, self.ssl_patch_size)  # [batch_size, 3, num_patches=hw, patch_size_h, patch_size_w]
        ssl_output = ssl_output.permute(0, 2, 3, 4, 1) # [batch_size, num_patches=hw, patch_size_h, patch_size_w, 3]
        
        _, num_patches, _, _, _ = ssl_output.shape
        ssl_output = ssl_output.reshape(batch_size, num_patches, -1) # [batch_size, num_patches, patch_size*patch_size*3]
        
        ssl_loss = self.ssl_criterion(ssl_output, batch['ssl_target']).mean(dim=2)
        ssl_pred = ssl_output

      else:
        raise ValueError()

      if self.ssl_loss_only_for_transformed:
        ssl_loss = (ssl_loss * batch['ssl_patches_mask']).sum() / batch['ssl_patches_mask'].sum()
      else:
        ssl_loss = ssl_loss.mean()

    else:
      ssl_loss = 0

    return loss, loss_dict, ssl_loss

  def training_step(self, batch, batch_idx):
    loss, loss_dict, ssl_loss = self.common_step(batch, batch_idx)     

    if self.ssl_task == "none":
      total_loss = loss
    else:
      total_loss = loss + self.ssl_loss_weight * ssl_loss

    self.log("train/loss", loss)
    for k,v in loss_dict.items():
      self.log("train/" + k, v.item())
    self.log("train/ssl_loss", ssl_loss)
    self.log("train/total_loss", total_loss)

    return total_loss

  def validation_step(self, batch, batch_idx):
    # get the inputs
    pixel_values = batch["pixel_values"].to(self.device)
    pixel_mask = batch["pixel_mask"].to(self.device)
    labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

    # forward pass
    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, return_dict=True)
    loss, loss_dict = outputs.loss, outputs.loss_dict

    # ssl loss
    if self.ssl_task != 'none':
      ssl_detr_outputs = self.model(pixel_values=batch['ssl_patches'], pixel_mask=pixel_mask, labels=labels, return_dict=True)
      encoder_output = ssl_detr_outputs.encoder_last_hidden_state # [batch_size, num_patches, hidden_dim]
      
      if self.ssl_task == 'jigsaw-discrete' or self.ssl_task == 'mim-discrete':
        ssl_output = self.ssl_pre_prediction_head(encoder_output) # [batch_size, num_patches, hidden_dim]
        ssl_output = self.ssl_prediction_head(ssl_output) # [batch_size, num_patches, num_classes]
        ssl_output = ssl_output.permute(0, 2, 1) # [batch_size, num_patches, num_classes] -> [batch_size, num_classes, num_patches]
        ssl_loss = self.ssl_criterion(ssl_output, batch['ssl_target'])
        ssl_pred = torch.argmax(ssl_output, dim=1) # [batch_size, num_patches]

      elif self.ssl_task == 'jigsaw-continuous' or self.ssl_task == 'mim-continuous':
        #ssl_output = self.ssl_prediction_head(encoder_output) # [batch_size, num_patches, num_pixel_locations]
        ssl_output = self.ssl_pre_prediction_head(encoder_output) # [batch_size, num_patches, hidden_dim]
        ssl_output = ssl_output.transpose(1, 2) # [batch_size, hidden_dim, num_patches]
        num_patch_rows = int(math.sqrt(ssl_output.size()[2]))
        ssl_output = ssl_output.unflatten(2, (num_patch_rows, num_patch_rows)) # [batch_size, hidden_dim, num_patch_rows(or)h, num_patch_cols(or)w]
        ssl_output = self.ssl_prediction_head(ssl_output) # [batch_size, 3, H, W] 

        batch_size, C, H, W = ssl_output.shape
        ssl_output = ssl_output.unfold(2, self.ssl_patch_size, self.ssl_patch_size).unfold(3, self.ssl_patch_size, self.ssl_patch_size) # [batch_size, 3, h, w, patch_size_h, patch_size_w]
        ssl_output = ssl_output.contiguous().view(batch_size, C, -1, self.ssl_patch_size, self.ssl_patch_size)  # [batch_size, 3, num_patches=hw, patch_size_h, patch_size_w]
        ssl_output = ssl_output.permute(0, 2, 3, 4, 1) # [batch_size, num_patches=hw, patch_size_h, patch_size_w, 3]
        
        _, num_patches, _, _, _ = ssl_output.shape
        ssl_output = ssl_output.reshape(batch_size, num_patches, -1) # [batch_size, num_patches, patch_size*patch_size*3]

        ssl_loss = self.ssl_criterion(ssl_output, batch['ssl_target']).mean(dim=2) # [batch_size, num_patches]
        ssl_pred = ssl_output

      else:
        raise ValueError()

      if self.ssl_loss_only_for_transformed:
        ssl_loss = (ssl_loss * batch['ssl_patches_mask']).sum() / batch['ssl_patches_mask'].sum()
      else:
        ssl_loss = ssl_loss.mean()

    else:
      ssl_loss = 0

    if self.ssl_task == "none":
      total_loss = loss
    else:
      total_loss = loss + self.ssl_loss_weight * ssl_loss

    # logging
    self.log("validation/loss", loss, sync_dist=True)
    for k,v in loss_dict.items():
      self.log("validation/" + k, v.item(), sync_dist=True)
    self.log("validation/ssl_loss", ssl_loss)
    self.log("validation/total_loss", total_loss)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    self.coco_evaluator.update(res)

    # ssl evaluation
    if self.ssl_task.endswith('discrete'):
      if self.ssl_loss_only_for_transformed:
        mask = batch['ssl_patches_mask'] == 1
        ssl_acc = self.ssl_metric(ssl_pred[mask], batch['ssl_target'][mask])
      else:
        ssl_acc = self.ssl_metric(ssl_pred, batch['ssl_target'])
      self.log("validation/ssl_acc", ssl_acc)
    elif self.ssl_task.endswith('continuous') and batch_idx == 0:
      max_images = 8

      input_imgs = self.inv_trans(batch['ssl_patches'][:max_images]) # [batch_size, 3, H, W]
      self.logger.log_image("images/ssl_input", [x for x in input_imgs])

      pred_imgs = ssl_pred[:max_images] # [batch_size, num_patches, patch_size*patch_size*3]
      _, num_patches, num_patch_pixels = pred_imgs.shape
      patch_size = int((num_patch_pixels // 3)**0.5)
      height = int(num_patches**0.5)
      pred_imgs = pred_imgs.reshape(max_images, num_patches, patch_size, patch_size, 3) # [batch_size, num_patches, patch_h, patch_w, 3]
      pred_imgs = pred_imgs.reshape(max_images, height, height , patch_size, patch_size, 3) # [batch_size, h, w, patch_h, patch_w, 3]
      pred_imgs = pred_imgs.permute(0, 5, 1, 3, 2, 4) # [batch_size, 3, h, patch_h, w, patch_w]
      pred_imgs = pred_imgs.reshape(max_images, 3, patch_size*height, patch_size*height) # [batch_size, 3, H, W]
      pred_imgs = self.inv_trans(pred_imgs)
      self.logger.log_image("images/ssl_pred", [x for x in pred_imgs])

    return total_loss

  def validation_epoch_end(self, validation_step_outputs):
    self.coco_evaluator.synchronize_between_processes()
    self.coco_evaluator.accumulate()
    self.coco_evaluator.summarize()
    results = self.coco_evaluator.coco_eval['bbox'].stats
    self.log("result/all_ap_50_95", results[0], sync_dist=True)
    self.log("result/all_ap_50", results[1], sync_dist=True)
    self.log("result/all_ap_75", results[2], sync_dist=True)
    self.log("result/small_ap_50_95", results[3], sync_dist=True)
    self.log("result/medium_ap_50_95", results[4], sync_dist=True)
    self.log("result/large_ap_50_95", results[5], sync_dist=True)
    self.log("result/all_ar_50_95_1d", results[6], sync_dist=True)
    self.log("result/all_ar_50_95_10d", results[7], sync_dist=True)
    self.log("result/all_ar_75_95", results[8], sync_dist=True)
    self.log("result/small_ar_50_95", results[9], sync_dist=True)
    self.log("result/medium_ar_50_95", results[10], sync_dist=True)
    self.log("result/large_ar_50_95", results[11], sync_dist=True)

    self.coco_evaluator = CocoEvaluator(self.dataset_val_coco, ['bbox'])

    if self.ssl_task.endswith('discrete'):
      ssl_acc = self.ssl_metric.compute()
      self.log("result/ssl_acc", ssl_acc)
      self.ssl_metric.reset()

  def configure_optimizers(self):
    param_dicts = [
      {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
      {
        "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": self.lr_backbone,
      },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
    
    return optimizer


def create_model(args, dataset_val_coco, feature_extractor):
  model = Detr(args=args, dataset_val_coco=dataset_val_coco, feature_extractor = feature_extractor)

  return model