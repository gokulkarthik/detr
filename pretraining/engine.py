import copy
from unittest import result
from numpy import mat
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Accuracy
from transformers import DetrConfig, DetrForObjectDetection, DetrModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class DetrEncoder(DetrModel):
    """
    src: https://github.com/huggingface/transformers/src/transformers/models/detr/modeling_detr.py#L1170
    """

    def __init__(self, args):
        super().__init__(DetrConfig())
        self.decoder = None
        self.query_position_embeddings = None

    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import DetrFeatureExtractor, DetrModel
        >>> from PIL import Image
        >>> import requests
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        assert mask is not None, "Backbone does not return downsampled pixel mask"

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        position_embeddings = position_embeddings_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        encoder_outputs = self.encoder(
            inputs_embeds=flattened_features,
            attention_mask=flattened_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        N, HW, C = encoder_outputs[0].shape
        output = encoder_outputs[0] # [batch_size, num_patches, hidden_dim]

        return output



class DetrEncoderforSSL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.input_image_size = args.max_image_size
                                              
        self.model = DetrEncoder(args)
        if args.backbone_pretrained:
            source_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)
            self.model.backbone.load_state_dict(source_model.model.backbone.state_dict())
            del source_model
        if args.backbone_freeze:
            for n, p in self.model.backbone.named_parameters():
                p.requires_grad_(False)

        if args.pretext_task == 'jigsaw-discrete':
            self.model.prediction_head = nn.Linear(self.model.config.d_model, (self.input_image_size//32)**2)
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.metric = Accuracy()
        elif args.pretext_task == 'mim-discrete':
            self.model.prediction_head = nn.Linear(self.model.config.d_model, 8192)
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.metric = Accuracy()
        elif args.pretext_task == 'mim-continuous' or args.pretext_task == 'jigsaw-continuous':
            self.model.prediction_head = nn.Linear(self.model.config.d_model, 32*32*3)
            self.criterion = nn.L1Loss(reduction='none')

        self.pretext_task = args.pretext_task
        self.loss_only_for_transformed = args.loss_only_for_transformed
        self.lr = args.lr
        self.lr_backbone = args.lr_backbone
        self.weight_decay = args.weight_decay

    def forward(self, batch):
        encoder_output = self.model.forward(pixel_values=batch['patches']) # [batch_size, num_patches, hidden_dim]
        output = self.model.prediction_head(encoder_output) # [batch_size, num_patches, num_classes/num_pixel_locations]
        if self.pretext_task == 'jigsaw-discrete' or self.pretext_task == 'mim-discrete':
            output = output.permute(0, 2, 1) # [batch_size, num_patches, num_classes] -> [batch_size, num_classes, num_patches]
            pred = torch.argmax(output, dim=1) # [batch_size, num_patches]
        elif self.pretext_task == 'jigsaw-continuous' or self.pretext_task == 'mim-continuous':
            output = F.sigmoid(output) # [batch_size, num_patches, patch_size*patch_size*3]
            pred = output # [batch_size, num_patches, patch_size*patch_size*3]

        return pred
     
    def common_step(self, batch):
        encoder_output = self.model.forward(pixel_values=batch['patches']) # [batch_size, num_patches, hidden_dim]
        output = self.model.prediction_head(encoder_output) # [batch_size, num_patches, num_classes/num_pixel_locations]

        if self.pretext_task == 'jigsaw-discrete' or self.pretext_task == 'mim-discrete':
            output = output.permute(0, 2, 1) # [batch_size, num_patches, num_classes] -> [batch_size, num_classes, num_patches]
            loss = self.criterion(output, batch['target'])
            pred = torch.argmax(output, dim=1) # [batch_size, num_patches]
        elif self.pretext_task == 'jigsaw-continuous' or self.pretext_task == 'mim-continuous':
            output = F.sigmoid(output) # [batch_size, num_patches, patch_size*patch_size*3]
            loss = self.criterion(output, batch['target']).mean(dim=2)
            pred = output
        else:
            raise ValueError()

        if self.pretext_task.startswith('jigsaw'):
            mask = (batch['patches_info'] == torch.arange(batch['patches_info'].shape[1], device=batch['patches_info'].device).long().unsqueeze(0)).long()
        elif self.pretext_task.startswith('mim'):
            mask = (batch['patches_info'] == 1).long()
        else:
            raise ValueError()
        
        loss = (loss * mask).sum() / mask.sum()

        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch)     
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)     
        self.log("val/loss", loss)
    
        if self.pretext_task.endswith('discrete'):
            acc = self.metric(pred, batch['target'])
            self.log("val/accuracy", acc)

        return loss

    def validation_epoch_end(self, outputs):
        if self.pretext_task.endswith('discrete'):
            acc = self.metric.compute()
            self.log("val/accuracy_end", acc)
            self.metric.reset()
        return super().validation_epoch_end(outputs)

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



def create_model(args):
    model = DetrEncoderforSSL(args=args)

    return model