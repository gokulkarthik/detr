import copy
from unittest import result
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch

from utils import CocoEvaluator

class Detr(pl.LightningModule):

     def __init__(self, args, dataset_val_coco, feature_extractor):
      super().__init__()
                                              
      if args.mscoco_pretrained:
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                          num_labels=args.num_labels,
                                                          ignore_mismatched_sizes=True)
      else:
        self.source_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                          num_labels=args.num_labels,
                                                          ignore_mismatched_sizes=True)
        self.model = DetrForObjectDetection(DetrConfig())
        self.model.config.num_labels = args.num_labels
        self.model.class_labels_classifier = copy.deepcopy(self.source_model.class_labels_classifier)
        self.model.model.backbone.load_state_dict(self.source_model.model.backbone.state_dict())
        self.source_model = None
        
      if args.encoder_init_ckpt != 'none':
        self.model.model.encoder.load_state_dict(torch.load(args.encoder_init_ckpt))

      if args.freeze_backbone:
        for n, p in self.model.named_parameters():
          if "backbone" in n:
            p.requires_grad_(False)

      if args.freeze_encoder:
        for n, p in self.model.named_parameters():
          if "encoder" in n:
            p.requires_grad_(False)

      self.lr = args.lr
      self.lr_backbone = args.lr_backbone
      self.weight_decay = args.weight_decay

      self.dataset_val_coco = dataset_val_coco
      self.coco_evaluator = CocoEvaluator(self.dataset_val_coco, ['bbox'])
      self.feature_extractor = feature_extractor

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("train/loss", loss)
        for k,v in loss_dict.items():
          self.log("train/" + k, v.item())

        return loss

    #  def validation_step_old(self, batch, batch_idx):
    #     loss, loss_dict = self.common_step(batch, batch_idx)     
    #     self.log("validation/loss", loss)
    #     for k,v in loss_dict.items():
    #       self.log("validation/" + k, v.item())

    #     return loss

     def validation_step(self, batch, batch_idx):
      # get the inputs
      pixel_values = batch["pixel_values"].to(self.device)
      pixel_mask = batch["pixel_mask"].to(self.device)
      labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

      # forward pass
      outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

      loss, loss_dict = outputs.loss, outputs.loss_dict
      self.log("validation/loss", loss)
      for k,v in loss_dict.items():
        self.log("validation/" + k, v.item())

      orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
      results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
      res = {target['image_id'].item(): output for target, output in zip(labels, results)}
      self.coco_evaluator.update(res)

      return loss

     def validation_epoch_end(self, validation_step_outputs):
      self.coco_evaluator.synchronize_between_processes()
      self.coco_evaluator.accumulate()
      self.coco_evaluator.summarize()
      results = self.coco_evaluator.coco_eval['bbox'].stats
      self.log("result/all_ap_50_95", results[0])
      self.log("result/all_ap_50", results[1])
      self.log("result/all_ap_75", results[2])
      self.log("result/small_ap_50_95", results[3])
      self.log("result/medium_ap_50_95", results[4])
      self.log("result/large_ap_50_95", results[5])
      self.log("result/all_ar_50_95_1d", results[6])
      self.log("result/all_ar_50_95_10d", results[7])
      self.log("result/all_ar_75_95", results[8])
      self.log("result/small_ar_50_95", results[9])
      self.log("result/medium_ar_50_95", results[10])
      self.log("result/large_ar_50_95", results[11])

      self.coco_evaluator = CocoEvaluator(self.dataset_val_coco, ['bbox'])

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