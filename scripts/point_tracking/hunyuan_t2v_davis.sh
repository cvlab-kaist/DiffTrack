model=hunyuan_t2v
python evaluate_tapvid.py \
    --model $model \
    --matching_layer 16 --matching_timestep 29 --inverse_step 29 \
    --output_dir ./output \
    --eval_dataset davis_first --tapvid_root /path/to/data \
    --resize_h 480 --resize_w 720 \
    --chunk_frame_interval --average_overlapped_corr \
    --pipe_device cuda:0 \
    --vis_video --tracks_leave_trace 15