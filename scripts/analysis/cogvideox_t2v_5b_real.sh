python analyze_real.py \
    --output_dir ./output \
    --model cogvideox_t2v_5b --num_inference_steps 50 \
    --matching_accuracy --conf_attn_score \
    --resize_h 480 --resize_w 720 \
    --eval_dataset davis_first --tapvid_root /path/to/data \
    --device cuda:0
