import argparse
import os
import glob
import random
import numpy as np
import cv2
import torch

from itertools import product
from PIL import Image

from diffusers import (
    CogVideoXPipeline, 
    CogVideoXImageToVideoPipeline, 
    CogVideoXImageToVideoPipeline2B, 
    HunyuanVideoTransformer3DModel, 
    HunyuanVideoPipeline
)

from utils.confidence_attention_score import ConfidenceAttentionScore
from utils.evaluation import MatchingEvaluator
from utils.query_key_vis import QueryKeyVisualizer, src_pos_img
from utils.track_vis import Visualizer
from utils.aggregate_results import save_accuracy_mean, save_score_mean
from utils.harmonic_mean import save_harmonic_mean


def load_pipe(model, device):
    if model == "cogvideox_t2v_5b":
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_t2v_2b":
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_i2v_5b":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_i2v_2b":
        pipe = CogVideoXImageToVideoPipeline2B.from_pretrained(
            "NimVideo/cogvideox-2b-img2vid",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "hunyuan_t2v":
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, transformer=transformer, 
            torch_dtype=torch.float16
        ).to(device)
    else:
        raise ValueError("Select model in ['cogvideox_t2v_5b', 'cogvideox_t2v_2b', 'cogvideox_i2v_2b', 'cogvideox_i2v_5b', 'hunyuan_t2v']")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe


def main(args):

    device = args.device

    output_dir = args.output_dir
    prompt_path = args.txt_path

    params = {
        'query_coords': None,
        'video_mode': args.video_mode,
        'trajectory': args.matching_accuracy or args.vis_track,
        'attn_weight': args.conf_attn_score,

        'matching_layer': args.vis_layers,
        'query_key': False,
        'feature': False,
    }
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipe(model=args.model, device=device)


    with open(prompt_path, "r") as file:
        prompts = file.readlines()
    

    for i, prompt in enumerate(prompts):
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        generator = torch.manual_seed(seed)
        
        save_dir = os.path.join(output_dir, f'{i:03d}')
        os.makedirs(save_dir, exist_ok=True)

        prompt = prompt.strip()
        gt_track = torch.from_numpy(np.load(os.path.join(args.track_dir, f'{i:03d}.npy'))).to(args.device)
        gt_visibility = torch.from_numpy(np.load(os.path.join(args.visibility_dir, f'{i:03d}.npy'))).to(args.device)

        layer_num = pipe.transformer.config.num_layers 
        if args.model == 'hunyuan_t2v':
            layer_num += pipe.transformer.config.num_single_layers

        if args.conf_attn_score:
            conf_attn_score = ConfidenceAttentionScore(
                num_inference_steps=args.num_inference_steps,
                num_layers=layer_num,
                visibility=gt_visibility,
                model=args.model
            )
        else:
            conf_attn_score = None


        if args.matching_accuracy:
            qk_acc_evaluator = MatchingEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=layer_num,
                gt_tracks=gt_track,
                gt_visibility=gt_visibility
            )
            feat_acc_evaluator = MatchingEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=layer_num,
                gt_tracks=gt_track,
                gt_visibility=gt_visibility
            )
        else:
            qk_acc_evaluator = None
            feat_acc_evaluator = None


        if args.vis_attn_map:
            querykey_visualizer = QueryKeyVisualizer(
                save_timestep_idxs=args.vis_timesteps,
                save_layers=args.vis_layers,
                output_dir=save_dir,
                model=args.model
            )
        else:
            querykey_visualizer = None


        with torch.no_grad(): 
            if "i2v" in args.model:
                image = Image.open(os.path.join(args.image_dir, f'{i:03d}.png'))

                video, attn_query_keys, features, vis_trajectory = pipe(
                    image=image,
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    conf_attn_score=conf_attn_score,
                    qk_acc_evaluator=qk_acc_evaluator,
                    feat_acc_evaluator=feat_acc_evaluator,
                    querykey_visualizer=querykey_visualizer,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params
                )
            else:
                video, attn_query_keys, features, vis_trajectory = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    conf_attn_score=conf_attn_score,
                    qk_acc_evaluator=qk_acc_evaluator,
                    feat_acc_evaluator=feat_acc_evaluator,
                    querykey_visualizer=querykey_visualizer,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params
                )

            if conf_attn_score is not None:
                conf_attn_score.report(save_dir)
                print(f"Affinity score saved at {save_dir}")
            
            if qk_acc_evaluator is not None:
                qk_acc_evaluator.report(os.path.join(save_dir, 'qk_acc.txt'))
                print(f"Matching accuracy saved at {os.path.join(save_dir, 'qk_acc.txt')}")
            if feat_acc_evaluator is not None:
                feat_acc_evaluator.report(os.path.join(save_dir, 'feat_acc.txt'))
                print(f"Matching accuracy saved at {os.path.join(save_dir, 'feat_acc.txt')}")
            

            if querykey_visualizer is not None:
                # Visualize PCA of query-key
                q_pca, k_pca = querykey_visualizer.save_pcas(attn_query_keys=attn_query_keys)
                pca_dir = os.path.join(save_dir, 'pca')
                q_pca_dir = os.path.join(pca_dir, 'query')
                k_pca_dir = os.path.join(pca_dir, 'key')

                os.makedirs(q_pca_dir, exist_ok=True)
                os.makedirs(k_pca_dir, exist_ok=True)
                for t, timestep in enumerate(args.vis_timesteps):
                    for l, layer in enumerate(args.vis_layers):
                        os.makedirs(os.path.join(q_pca_dir, f't{timestep}_l{layer}'), exist_ok=True)
                        os.makedirs(os.path.join(k_pca_dir, f't{timestep}_l{layer}'), exist_ok=True)
                        for f in range(13):
                            Image.fromarray(q_pca[l][t][f]).save(os.path.join(q_pca_dir, f't{timestep}_l{layer}', f'{f}.png'))
                            Image.fromarray(k_pca[l][t][f]).save(os.path.join(k_pca_dir, f't{timestep}_l{layer}', f'{f}.png'))

                # Visualize cross attention map (query-key / feature)
                for pos_h, pos_w in list(product(args.pos_h, args.pos_w)):
                    attention_dir = os.path.join(save_dir, 'attention_map', f'({pos_h},{pos_w})')
                    os.makedirs(attention_dir, exist_ok=True)

                    pos_img = src_pos_img(video[0][0], pos_h, pos_w)
                    pos_img.save(os.path.join(attention_dir, 'src_pos.png'))

                    qk_attns = querykey_visualizer.save_i2i_attn_map(
                        attn_query_keys=attn_query_keys,
                        pos=(pos_h, pos_w),
                        mode='qk'
                    )
                    feat_attns = querykey_visualizer.save_i2i_attn_map(
                        attn_query_keys=features,
                        pos=(pos_h, pos_w),
                        mode='feature'
                    )
                    qk_attn_dir = os.path.join(attention_dir, 'qk')
                    feat_attn_dir = os.path.join(attention_dir, 'feature')


                    os.makedirs(qk_attn_dir, exist_ok=True)
                    os.makedirs(feat_attn_dir, exist_ok=True)
                    for t, timestep in enumerate(args.vis_timesteps):
                        for l, layer in enumerate(args.vis_layers):
                            os.makedirs(os.path.join(qk_attn_dir, f't{timestep}_l{layer}'), exist_ok=True)
                            os.makedirs(os.path.join(feat_attn_dir, f't{timestep}_l{layer}'), exist_ok=True)
                            for f in range(13):
                                Image.fromarray(qk_attns[l][t][f]).save(os.path.join(qk_attn_dir, f't{timestep}_l{layer}', f'{f}.png'))
                                Image.fromarray(feat_attns[l][t][f]).save(os.path.join(feat_attn_dir, f't{timestep}_l{layer}', f'{f}.png'))

            if args.vis_track:
                track_dir = os.path.join(save_dir, 'track')
                os.makedirs(track_dir, exist_ok=True)

                valid_mask = gt_visibility[0,0,:]
                first_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
                vis = Visualizer(save_dir=track_dir, pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=15, fps=8)
                vis.visualize(
                    video=video * 255,
                    tracks=vis_trajectory[:, :, first_idx, :],
                    filename="video.mp4", 
                    query_frame=0
                )


    if args.matching_accuracy:
        save_accuracy_mean(file_list=glob.glob(os.path.join(output_dir, '*/qk_acc.txt')), output_path=os.path.join(output_dir, 'total_qk_acc.csv'))
        save_accuracy_mean(file_list=glob.glob(os.path.join(output_dir, '*/feat_acc.txt')), output_path=os.path.join(output_dir, 'total_feat_acc.csv'))

    if args.conf_attn_score:
        save_score_mean(file_list=glob.glob(os.path.join(output_dir, '*/confidence_score.xlsx')), output_path=os.path.join(output_dir, 'total_confidence_score.xlsx'))
        save_score_mean(file_list=glob.glob(os.path.join(output_dir, '*/attention_score.xlsx')), output_path=os.path.join(output_dir, 'total_attention_score.xlsx'))

    if args.matching_accuracy and args.conf_attn_score:
        save_harmonic_mean(
            acc_path=os.path.join(output_dir, 'total_qk_acc.csv'), 
            conf_path=os.path.join(output_dir, 'total_confidence_score.xlsx'),
            attn_path=os.path.join(output_dir, 'total_attention_score.xlsx'),
            output_path=os.path.join(output_dir, 'harmonic_mean.csv')
        )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        choices=[
            "cogvideox_t2v_5b", "cogvideox_t2v_2b", "cogvideox_i2v_5b", "cogvideox_i2v_2b", "hunyuan_t2v"], 
        default='cogvideox_t2v_2b'
    )
    parser.add_argument("--video_mode", type=str, choices=["fg", "bg"], default='fg')
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)


    parser.add_argument("--conf_attn_score", action='store_true')
    parser.add_argument("--matching_accuracy", action='store_true')

    parser.add_argument("--vis_timesteps", nargs='+', type=int, default=[49])
    parser.add_argument("--vis_layers", nargs='+', type=int, default=[17])

    parser.add_argument("--vis_attn_map", action='store_true')
    parser.add_argument("--pos_h", type=int, nargs='+', default=[16])
    parser.add_argument("--pos_w", type=int, nargs='+', default=[16])

    parser.add_argument("--vis_track", action='store_true')

    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--track_dir", type=str, required=True)
    parser.add_argument("--visibility_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default='cuda:0')
    
    args = parser.parse_args()

    main(args)
