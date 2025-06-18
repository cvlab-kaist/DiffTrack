import argparse
import torch
from diffusers import CogVideoXPipeline_PAG
from diffusers.utils import export_to_video
import os
import random
import numpy as np

def load_pipe(model_version, pag_layers, device):
    if model_version == '5b':
        pipe = CogVideoXPipeline_PAG.from_pretrained(
            "THUDM/CogVideoX-5b",
            pag_applied_layers=[
                f"transformer_blocks.{layer}" for layer in pag_layers
            ], 
            torch_dtype=torch.bfloat16
        ).to(device)
    else:
        pipe = CogVideoXPipeline_PAG.from_pretrained(
            "THUDM/CogVideoX-2b",
            pag_applied_layers=[
                f"transformer_blocks.{layer}" for layer in pag_layers
            ], 
            torch_dtype=torch.bfloat16
        ).to(device)

    return pipe

def main(args):

    device = args.device
    output_dir = args.output_dir
    prompt_path = args.txt_path 

    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipe(model_version=args.model_version, pag_layers=args.pag_layers, device=device)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    params = {
        'trajectory': False,
        'attn_weight': False,
        'query_key': False,
        'head_matching_layer': -1,
        'matching_layer': [],
    }

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

        prompt = prompt.strip()

        with torch.no_grad(): 
            
            video = pipe(
                prompt=prompt,
                height=480,
                width=720,
                num_frames=49,
                num_inference_steps=args.num_inference_steps,
                return_dict=False,
                generator=generator,
                output_type="pt",
                params=params,
                pag_scale=args.pag_scale,
                guidance_scale=args.cfg_scale
            ).frames

            video_np = video.squeeze(0).to(torch.float32).permute(0, 2, 3, 1).cpu().numpy()
            export_to_video(video_np, os.path.join(output_dir, f"video_{i}.mp4"), fps=8)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--model_version", type=str, default='2b')
    parser.add_argument("--pag_scale", type=float, default=0.4, help="Scale for CAG attention visualization")
    parser.add_argument("--cfg_scale", type=float, default=6, help="Scale for CFG attention visualization")
    parser.add_argument("--pag_layers", type=int, nargs='+')
    parser.add_argument("--pag_timestep", type=float, default=float('inf'))

    args = parser.parse_args()


    main(args)
