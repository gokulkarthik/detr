python3 main.py --dataset isaid \
    --min_image_size 512 \
    --max_image_size 512 \
    --mscoco_pretrained f \
    --freeze_backbone f \
    --freeze_encoder f \
    --encoder_init_ckpt models/devout-disco-44-mim-continuous-ttt.ckpt \
    --gpus '0 1 2 3' \
    --strategy 'ddp' \
    --precision 32 \
    --batch_size 64 \
    --max_epochs 100 