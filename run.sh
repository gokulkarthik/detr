python3 main.py --dataset isaid \
    --min_image_size 1024 \
    --max_image_size 1024 \
    --mscoco_pretrained f \
    --freeze_backbone t \
    --freeze_encoder f \
    --encoder_init_ckpt none \
    --batch_size 16 \
    --max_epochs 10 