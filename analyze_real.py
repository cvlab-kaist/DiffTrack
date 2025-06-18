import argparse
import os
import glob
import random
import numpy as np
import torch

from diffusers import CogVideoXPipeline

from utils.confidence_attention_score import ConfidenceAttentionScore
from utils.evaluation import MatchingEvaluator
from utils.aggregate_results import score_mean, accuracy_mean
from utils.tapvid import TAPVid



def main(args):

    device = args.device

    output_dir = args.output_dir

    params = {
        'trajectory': args.matching_accuracy,
        'attn_weight': args.conf_attn_score,
        'query_key': False,
        'feature': False,
        'video_mode': '',
        'matching_layer': args.vis_layers,
        'query_coords': None,
    }
    os.makedirs(output_dir, exist_ok=True)

    dataset = TAPVid(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    model_id = "THUDM/CogVideoX-5b" if args.model == "cogvideox_t2v_5b" else "THUDM/CogVideoX-2b"
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


    for j, (input_video, gt_trajectory, gt_visibility, query_points_i, video_ori) in enumerate(dataloader):
        
        if j > 1: break
        valid_mask = (query_points_i[:,:,0] == 0)
        if not torch.any(valid_mask):
            continue
        _, first_idx = torch.nonzero(valid_mask, as_tuple=True)
        query_points_i = query_points_i[:, first_idx, :].to(device=device)
        gt_trajectory = gt_trajectory[:, :, first_idx, :].to(device=device)
        gt_visibility = gt_visibility[:, :, first_idx].to(device=device)

        if input_video.size(1) < args.video_max_len:
            continue

        if os.path.isdir(os.path.join(output_dir, f'{j:03d}')):
            continue
        
        if args.video_max_len != -1 and args.video_max_len < input_video.size(1):
            input_video = input_video[:,:args.video_max_len, ...]
            gt_trajectory = gt_trajectory[:, :args.video_max_len, :, :]
            gt_visibility = gt_visibility[:, :args.video_max_len, :]
        
        _, video_len, _, H, W = input_video.shape
        params['query_coords'] = query_points_i[...,1:]


        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        generator = torch.manual_seed(seed)
        
        save_dir = os.path.join(output_dir, f'{j:03d}')
        os.makedirs(save_dir, exist_ok=True)

        if args.conf_attn_score:
            conf_attn_score = ConfidenceAttentionScore(
                num_inference_steps=args.num_inference_steps,
                num_layers=pipe.transformer.config.num_layers,
                visibility=gt_visibility,
                model=args.model
            )
        else:
            conf_attn_score = None

        
        if args.matching_accuracy:
            layer_num = pipe.transformer.config.num_layers
            gt_tracks = gt_trajectory.clone()
            gt_tracks[..., 0] *= (W / 256)
            gt_tracks[..., 1] *= (H / 256)

            qk_acc_evaluator = MatchingEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=layer_num,
                gt_tracks=gt_tracks,
                gt_visibility=gt_visibility
            )
            feat_acc_evaluator = MatchingEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=layer_num,
                gt_tracks=gt_tracks,
                gt_visibility=gt_visibility
            )
        else:
            qk_acc_evaluator = None
            feat_acc_evaluator = None

        for inverse_timestep in range(50):
            with torch.no_grad():
                video, attn_query_keys, features, vis_trajectory = pipe(
                    prompt="",
                    height=480,
                    width=720,
                    num_frames=49,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    conf_attn_score=conf_attn_score,
                    qk_acc_evaluator=qk_acc_evaluator,
                    feat_acc_evaluator=feat_acc_evaluator,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params,
                    video=input_video,
                    inverse_step=inverse_timestep
                )

        if conf_attn_score is not None:
            conf_attn_score.report(save_dir)
            print(f"Affinity score saved at {save_dir}")
        
        if qk_acc_evaluator is not None:
            qk_acc_evaluator.report(os.path.join(save_dir, 'qk_acc.txt'))
            print(f"PCK saved at {os.path.join(save_dir, 'qk_acc.txt')}")
        if feat_acc_evaluator is not None:
            feat_acc_evaluator.report(os.path.join(save_dir, 'feat_acc.txt'))
            print(f"PCK saved at {os.path.join(save_dir, 'feat_acc.txt')}")
        

    if args.pck:
        accuracy_mean(file_list=glob.glob(os.path.join(output_dir, '*/qk_acc.txt')), output_path=os.path.join(output_dir, 'total_qk_cc.csv'))
        accuracy_mean(file_list=glob.glob(os.path.join(output_dir, '*/feat_acc.txt')), output_path=os.path.join(output_dir, 'total_feat_acc.csv'))

    if args.conf_attn_score:
        score_mean(file_list=glob.glob(os.path.join(output_dir, f'*/confidence_score.xlsx')), output_path=os.path.join(output_dir, f'total_confidence_score.xlsx'))
        score_mean(file_list=glob.glob(os.path.join(output_dir, f'*/attention_score.xlsx')), output_path=os.path.join(output_dir, f'total_attention_score.xlsx'))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["cogvideox_t2v_5b", "cogvideox_t2v_2b"], default='cogvideox_t2v_2b')
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--conf_attn_score", action='store_true')
    parser.add_argument("--matching_accuracy", action='store_true')

    parser.add_argument("--vis_timesteps", nargs='+', type=int, default=[49])
    parser.add_argument("--vis_layers", nargs='+', type=int, default=[17])

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)
    parser.add_argument("--video_max_len", type=int, default=49)

    args = parser.parse_args()


    main(args)
