


import torch
from diffusers import (
    CogVideoXTrackPipeline, 
    CogVideoXImageToVideoTrackPipeline, 
    CogVideoXImageToVideoTrackPipeline2B, 
    CogVideoXInversePipeline, 
    HunyuanVideoTransformer3DModel, 
    HunyuanVideoTrackPipeline,
)
from diffusers.schedulers import CogVideoXDDIMScheduler, UniPCMultistepScheduler
import os
import random
import numpy as np
import argparse
from utils.matching import corr_to_matches
import torch.nn.functional as F
from einops import rearrange
from utils.track_vis import Visualizer
from utils.evaluation import Evaluator, compute_tapvid_metrics
from utils.tapvid import TAPVid
from torchvision.transforms import ToPILImage
import math

PARAMS = {
    'trajectory': False,
    'attn_weight': False,
    'query_key': True,
    'head_matching_layer': -1
}

INFERENCE_STEP = {
    'cogvideox_t2v_5b': 50,
    'cogvideox_i2v_5b': 50,
    'cogvideox_t2v_2b': 50,
    'cogvideox_i2v_2b': 50,
    'wan': 50,
    'hunyuan_t2v': 30
}

TOTAL_LAYER_NUM = {
    'cogvideox_t2v_5b': 30,
    'cogvideox_i2v_5b': 30,
    'cogvideox_t2v_2b': 30,
    'cogvideox_i2v_2b': 30,
    'wan': 30,
    'hunyuan_t2v': 60
}


def process_chunk(
    chunk_video,
    generator,
    pipe,
    first_frame,
    matching_timestep,
    matching_layer,
    inverse_step,
    add_noise=False,
    inversion_pipe=None,
    model_name="cogvideox_t2v",
    prompt=""
):

    chunk_video = torch.cat([first_frame, chunk_video], dim=1) / 255.
    _, _, _, H, W = first_frame.shape 

    # Run the inversion and generation pipelines.
    with torch.no_grad():
        latents = None
        if inversion_pipe is not None:
            latents = inversion_pipe(
                height=H,
                width=W,
                prompt="",
                video=chunk_video,
                num_videos_per_prompt=1,
                num_inference_steps=INFERENCE_STEP[model_name],
                inverse_step=inverse_step,
                use_dynamic_cfg=True,
                guidance_scale=6,
                generator=generator,
                params=PARAMS
            )
        
        if "i2v" in model_name:
            to_pil = ToPILImage()
            frames, queries, keys, text_seq_length = pipe(
                image=to_pil(first_frame[0,0]),
                height=H,
                width=W,
                prompt=prompt,
                guidance_scale=6,
                num_inference_steps=INFERENCE_STEP[model_name],
                generator=generator,
                latents=latents,
                video=chunk_video,
                frame_as_latent=True,
                inverse_step=inverse_step,
                matching_timestep=matching_timestep,
                matching_layer=matching_layer,
                add_noise=add_noise,
                return_dict=False,
                params=PARAMS
            )
        else:
            frames, queries, keys, text_seq_length = pipe(
                height=H,
                width=W,
                prompt=prompt,
                guidance_scale=6,
                num_inference_steps=INFERENCE_STEP[model_name],
                generator=generator,
                latents=latents,
                video=chunk_video,
                frame_as_latent=True,
                inverse_step=inverse_step,
                matching_timestep=matching_timestep,
                matching_layer=matching_layer,
                add_noise=add_noise,
                return_dict=False,
                params=PARAMS
            )

    latent_scaling_size = 16
    h = H // latent_scaling_size
    w = W // latent_scaling_size

    f_num = queries[0][:, text_seq_length:].shape[1] // (h*w)  # 17550 // (30*45) => 13

    queries = torch.cat(queries, dim=-1)
    keys = torch.cat(keys, dim=-1)
   
    
    if "hunyuan" in model_name:
        query_frames = rearrange(queries[:, :-text_seq_length][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)
        key_frames = rearrange(keys[:, :-text_seq_length][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)
    else:
        query_frames = rearrange(queries[:, text_seq_length:][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)
        key_frames = rearrange(keys[:, text_seq_length:][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)

    return query_frames, key_frames

def combine_correlations(correlations, average_overlap=False):
    corr_dict = {}
    for frames_indices, corr in correlations:
        for i, frame_idx in enumerate(frames_indices):
            if frame_idx not in corr_dict:
                corr_dict[frame_idx] = [corr[i]]
            elif average_overlap:
                corr_dict[frame_idx].append(corr[i])
    combined_corr = [torch.mean(torch.stack(corr_dict[i], dim=0), dim=0) for i in sorted(corr_dict.keys())]
    return combined_corr


def load_pipe(model, device):
    if model == "cogvideox_t2v_5b":
        # Load the pipelines.
        pipe = CogVideoXTrackPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to(device=device, dtype=torch.bfloat16)
    elif model == "cogvideox_t2v_2b":
        # Load the pipelines.
        pipe = CogVideoXTrackPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to(device=device, dtype=torch.bfloat16)
    elif model == "cogvideox_i2v_5b":
        # Load the pipelines.
        pipe = CogVideoXImageToVideoTrackPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to(device=device, dtype=torch.bfloat16)
    elif model == "cogvideox_i2v_2b":
        pipe = CogVideoXImageToVideoTrackPipeline2B.from_pretrained(
            "NimVideo/cogvideox-2b-img2vid",
            torch_dtype=torch.bfloat16
        ).to(device=device)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif model == "hunyuan_t2v":
        do_inversion = False
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoTrackPipeline.from_pretrained(
            model_id, transformer=transformer, 
            torch_dtype=torch.bfloat16
        ).to(device=device)

    return pipe


def main(args):
    # Fix seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.manual_seed(seed)

    to_pil = ToPILImage()

    chunk_stride = args.chunk_stride if not args.chunk_frame_interval else 1
    inverse_step = args.inverse_step if args.inverse_step in range(INFERENCE_STEP[args.model]) else min(args.matching_timestep)
    
    assert inverse_step < INFERENCE_STEP[args.model], f"Inverse step should be smaller than inference step {INFERENCE_STEP[args.model]} for {args.model}"

    output_dir = os.path.join(args.output_dir, f'layer{args.matching_layer}_timestep{args.matching_timestep}_noise{args.add_noise}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)

    # Set up the dataset and evaluator.
    dataset = TAPVid(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset
    
    pipe = load_pipe(args.model, args.pipe_device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    assert not (args.do_inversion and args.add_noise), "Cannot inverse noise and add random noise at same time"
    inversion_pipe = None
    if args.do_inversion and "cogvideox" in args.model:
        inversion_pipe = CogVideoXInversePipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)
        inversion_pipe.scheduler = CogVideoXDDIMScheduler.from_config(inversion_pipe.scheduler.config, timestep_spacing="trailing")
        inversion_pipe.to(device=args.inv_pipe_device, dtype=torch.bfloat16)
        inversion_pipe.vae.enable_slicing()
        inversion_pipe.vae.enable_tiling()
    
    matching_timestep = args.matching_timestep
    matching_layer = args.matching_layer

    assert min(matching_timestep) >= inverse_step, "Matching timestep should be bigger than inverse step"
    assert max(matching_layer) < TOTAL_LAYER_NUM[args.model], f"Matching layer index should be smaller than total number of layers({TOTAL_LAYER_NUM[args.model]} for {args.model})"

    PARAMS['matching_layer'] = matching_layer

    for j, (video, gt_trajectory, visibility, query_points_i, video_ori) in enumerate(dataloader):
        valid_mask = (query_points_i[:,:,0] == 0)
        if not torch.any(valid_mask):
            continue
        _, first_idx = torch.nonzero(valid_mask, as_tuple=True)
        query_points_i = query_points_i[:, first_idx, :]
        gt_trajectory = gt_trajectory[:, :, first_idx, :]
        visibility = visibility[:, :, first_idx]
        
        if args.video_max_len != -1 and args.video_max_len < video.size(1):
            video = video[:,:args.video_max_len, ...]
            gt_trajectory = gt_trajectory[:, :args.video_max_len, :, :]
            visibility = visibility[:, :args.video_max_len, :]

        _, T, _, H, W = video.shape

        if args.vis_video:
            _, _, _, H_ori, W_ori = video_ori.shape
            gt_trajectory_vis = gt_trajectory.clone()
            gt_trajectory_vis[..., 0] *= (W_ori / 256)
            gt_trajectory_vis[..., 1] *= (H_ori / 256)
            vis = Visualizer(save_dir=os.path.join(output_dir, 'gt'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
            frames = vis.visualize(video=video_ori, tracks=gt_trajectory_vis, filename =f"{j:03d}.mp4", query_frame=0, visibility=visibility)
            
            gt_dir = os.path.join(output_dir, 'gt', f'{j:03d}')
            os.makedirs(gt_dir, exist_ok=True)
            for f, frame in enumerate(frames[0]):
                to_pil(frame).save(os.path.join(gt_dir, f'{f}.png'))

        latent_scaling_size = 16
        h = H // latent_scaling_size
        w = W // latent_scaling_size

        interval = T // (args.chunk_len - 1) if args.chunk_frame_interval else 1
        chunk_start, chunk_end = 1, (args.chunk_len - 2) * interval + 1

        visited_idx = set([])
        correlation_per_chunk = []
        while len(visited_idx) < T - 1:
            chunk_indices = list(range(chunk_start, chunk_end + 1, interval))

            # Extract the current chunk.
            chunk_video = video[:, torch.tensor(chunk_indices),...]
            visited_idx.update(chunk_indices)

            # Process the chunk.
            query_frames, key_frames = process_chunk(
                chunk_video=chunk_video,
                generator=generator,
                matching_timestep=args.matching_timestep, 
                matching_layer=args.matching_layer,
                inverse_step=inverse_step,
                pipe=pipe,
                first_frame=video[:,0:1,...],
                add_noise=args.add_noise,
                inversion_pipe=inversion_pipe,
                model_name=args.model,
                prompt=""
            )
            
            B, head_dim, qk_len, _, _ = query_frames.shape
            corr = []
            for k in range(1, qk_len):
                attn_tts = torch.einsum("b h i d, b h j d -> b h i j", query_frames[:, :, 0, :, :], key_frames[:, :, k, :, :]) / math.sqrt(head_dim)
                attn_stt = torch.einsum("b h i d, b h j d -> b h i j", query_frames[:, :, k, :, :], key_frames[:, :, 0, :, :]) / math.sqrt(head_dim)
                attn_tts = attn_tts.softmax(dim=-1)
                attn_stt = attn_stt.softmax(dim=-1)
                attn_tts = attn_tts.mean(1)
                attn_stt = attn_stt.mean(1)
                correlation_from_t_to_s = rearrange(attn_tts, 'b (h w) c -> b c h w', h=h, w=w) # head mean
                correlation_from_t_to_s_T = rearrange(attn_stt, 'b c (h w) -> b c h w', h=h, w=w)
                correlation_from_t_to_s = (correlation_from_t_to_s + correlation_from_t_to_s_T) / 2 # torch.Size([1, 1350, 30, 45])
                corr.append(correlation_from_t_to_s)
            correlation_per_chunk.append((chunk_indices, corr))

            chunk_start += chunk_stride
            chunk_end = min(chunk_start + (args.chunk_len - 2) * interval, T-1)
            
        correlation = combine_correlations(correlation_per_chunk, average_overlap=args.average_overlapped_corr)

        # Normalize the coordinate channels (assumes channel 1 and 2 are spatial coordinates)
        queried_coords_latent = query_points_i.clone().float()
        queried_coords_latent[:,:,1] = queried_coords_latent[:,:,1] / latent_scaling_size
        queried_coords_latent[:,:,2] = queried_coords_latent[:,:,2] / latent_scaling_size

        # Normalize coordinates for grid sampling.
        margin = W/(64*latent_scaling_size)
        norm_coords = queried_coords_latent[:,:,1:].clone()
        norm_coords[:, :, 0] = (norm_coords[:, :, 0] / (w - margin)) * 2 - 1.0
        norm_coords[:, :, 1] = (norm_coords[:, :, 1] / (h - margin)) * 2 - 1.0

        num_queries = query_points_i.size(1)
        grid = norm_coords.view(B, num_queries, 1, 2).to(args.pipe_device)

        # The initial track is the (normalized) query coordinates.
        tracks = []
        tracks.append(queried_coords_latent[:,:,1:].unsqueeze(1).to(args.pipe_device))

        # For each of 12 latents (or timesteps) compute the displacement.
        for k in range(len(correlation)):
            correlation_from_t_to_s = correlation[k].to(args.pipe_device) / (len(matching_timestep) * len(matching_layer))
            (x_source, y_source, x_target, y_target, score) = corr_to_matches(
                correlation_from_t_to_s.view(1, h, w, h, w).unsqueeze(1), get_maximum=True, do_softmax=True, device=args.pipe_device
            )
            mapping_set = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1)
            mapping_set = mapping_set.view(1, h, w, 2).permute(0, 3, 1, 2)
            track = F.grid_sample(mapping_set.to(correlation_from_t_to_s.device), grid=grid, mode='bilinear', align_corners=True)
            track = rearrange(track, "b c h w -> b () (h w) c")
            tracks.append(track)

        trajectory = torch.cat(tracks, dim=1)

        # Scale back the coordinates.
        scaling_factor_x = latent_scaling_size 
        scaling_factor_y = latent_scaling_size

        trajectory[..., 0] *= scaling_factor_x
        trajectory[..., 1] *= scaling_factor_y

        if args.vis_video:
            vis_traj = trajectory.clone()
            vis_traj[..., 0] *= (W_ori / W)
            vis_traj[..., 1] *= (H_ori / H)
            vis = Visualizer(save_dir=os.path.join(output_dir, 'pred'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
            frames = vis.visualize(video=video_ori, tracks=vis_traj, filename =f"{j:03d}.mp4", query_frame=0)
            
            pred_dir = os.path.join(output_dir, 'pred', f'{j:03d}')
            os.makedirs(pred_dir, exist_ok=True)
            for f, frame in enumerate(frames[0]):
                to_pil(frame).save(os.path.join(pred_dir, f'{f}.png'))

        query_points_np = query_points_i.clone().cpu().numpy()
        gt_tracks = gt_trajectory.clone().permute(0, 2, 1, 3)[:, :, :trajectory.shape[1], :].cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = trajectory.permute(0, 2, 1, 3).cpu().numpy()
        pred_tracks[...,0] *= (256 / W)
        pred_tracks[...,1] *= (256 / H)
        pred_occluded = torch.zeros_like(visibility).permute(0, 2, 1).cpu().numpy()
        
        out_metrics = compute_tapvid_metrics(query_points_np, gt_occluded, gt_tracks, pred_occluded, pred_tracks,"first" if queried_first else "strided")
        evaluator.update(out_metrics, T, log_file=os.path.join(output_dir, 'log.txt'))
        

    evaluator.report(log_file=os.path.join(output_dir, 'log.txt'))
    print(f"Evaluation result saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["wan", "cogvideox_t2v_5b", "cogvideox_t2v_2b", "cogvideox_i2v_5b", "cogvideox_i2v_2b", "hunyuan_t2v"], default='cogvideox_t2v_5b')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)

    parser.add_argument('--chunk_stride', type=int, default=12)
    parser.add_argument('--chunk_len', type=int, default=13)
    parser.add_argument('--chunk_frame_interval', action='store_true')
    parser.add_argument('--average_overlapped_corr', action='store_true')
    parser.add_argument("--video_max_len", type=int, default=-1)
    parser.add_argument("--do_inversion", action='store_true')
    parser.add_argument("--add_noise", action='store_true')

    parser.add_argument('--matching_layer', nargs='+', type=int)
    parser.add_argument("--matching_timestep", nargs='+', type=int)
    parser.add_argument("--inverse_step", type=int, default=None)
    
    parser.add_argument("--vis_video", action='store_true')
    parser.add_argument("--tracks_leave_trace", type=int, default=15)

    parser.add_argument("--pipe_device", type=str, default='cuda:0')
    parser.add_argument("--inv_pipe_device", type=str, default='cuda:0')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)

    args = parser.parse_args()

    main(args)
    