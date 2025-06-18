CUDA_VISIBLE_DEVICES=0 python motion_guidance.py \
    --output_dir ./output \
    --model_version 5b \
    --txt_path ./dataset/cag_prompts.txt \
    --pag_layers 15 17 18 \
    --pag_scale 1 \
    --cfg_scale 6