import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torchvision import transforms
from ...schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from .pipeline_cogvideox import CogVideoXPipeline, CogVideoXPipelineOutput, retrieve_timesteps

import os


def resize_for_crop(image, crop_h, crop_w):
    img_h, img_w = image.shape[-2:]
    if img_h >= crop_h and img_w >= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    elif img_h <= crop_h and img_w <= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w 
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def prepare_image(image, video_size, do_resize=True, do_crop=True):
    image_tensor = transforms.functional.to_tensor(image) * 2 - 1
    if do_resize:
        image_tensor = resize_for_crop(image_tensor, crop_h=video_size[0], crop_w=video_size[1])
    if do_crop:
        image_tensor = transforms.functional.center_crop(image_tensor, video_size)
    return image_tensor.unsqueeze(0).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()


class CogVideoXImageToVideoPipeline2B(CogVideoXPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        conf_attn_score=None,
        qk_acc_evaluator=None,
        feat_acc_evaluator=None,
        querykey_visualizer=None,
        vis_timesteps=None,
        vis_layers=None,
        params=None
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        
        # 5. Prepare latents.
        latent_channels = 16 #self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        ## 5.1 Prepare image
        start_frame = None
        if image is not None: 
            start_frame = prepare_image(image, (height, width))
            start_frame = start_frame.to(dtype=self.vae.dtype, device=self.vae.device)
            start_frame = start_frame.repeat(1, 1, num_frames, 1, 1)
            
            start_frame = self.vae.encode(start_frame).latent_dist.sample()
            
            start_frame = start_frame.permute(0, 2, 1, 3, 4).contiguous()
            start_frame = start_frame * self.vae.config.scaling_factor
            start_frame = torch.cat([start_frame] * 2) if do_classifier_free_guidance else start_frame

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        text_len = prompt_embeds.size(1)
        _, f_num, _, h, w = latents.shape
        h, w = h//2, w//2

        vis_trajectory = None
        if conf_attn_score is not None:
            conf_attn_score.reset(text_len=text_len, latent_h=h, latent_w=w, latent_f=f_num)

        attn_query_keys = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noisy_model_input = torch.cat([latent_model_input, start_frame], dim=2)

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    args=params
                )[0]
                noise_pred = noise_pred.float()


                for l, blk in enumerate(self.transformer.transformer_blocks):
                    if params['attn_weight']:
                        attn_weight = blk.attn1.processor.attn_weight
                        conf_attn_score.update(attn_weight=attn_weight, layer=l, timestep_idx=i)
                        del blk.attn1.processor.attn_weight
                    
                    if params['trajectory']:
                        if qk_acc_evaluator is not None:
                            trajectory = blk.attn1.processor.trajectory_qk
                            qk_acc_evaluator.update(pred_tracks=trajectory, layer=l, timestep_idx=i)
                            del blk.attn1.processor.trajectory_qk

                        if feat_acc_evaluator is not None:
                            trajectory = blk.attn1.processor.trajectory_feat
                            feat_acc_evaluator.update(pred_tracks=trajectory, layer=l, timestep_idx=i)
                            del blk.attn1.processor.trajectory_feat

                        if vis_timesteps[0] == i and vis_layers[0] == l:
                            vis_trajectory = trajectory

    
                attn_query_keys = []
                features = []
                if querykey_visualizer is not None:
                    if i in vis_timesteps:
                        queries_, keys_, feats_ = [], [], []
                        for l in vis_layers:
                            blk = self.transformer.transformer_blocks[l]
                            Q = blk.attn1.processor.query[1]
                            K = blk.attn1.processor.key[1]
                            feat = blk.attn1.processor.feature[1]
                            queries_.append(Q)
                            keys_.append(K)
                            feats_.append(feat)
                            del blk.attn1.processor.query
                            del blk.attn1.processor.key
                            del blk.attn1.processor.feature
                        attn_query_keys.append([queries_, keys_])
                        features.append(feats_)
                            
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

    
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        if querykey_visualizer is not None:
            querykey_visualizer.set_attributes(
                text_len=text_len, latent_h=h, latent_w=w, latent_f=f_num,
                timesteps=timesteps, frames=video[0].cpu().float())
        

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, attn_query_keys, features, vis_trajectory)

        return CogVideoXPipelineOutput(frames=video)