import torch
import torchvision
import os
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

from sklearn.decomposition import PCA


def src_pos_img(image, height, width):
    height_circle = int((height+0.5) / 14. * 224)
    width_circle = int((width+0.5) / 14. * 224)

    image = image.permute(1, 2, 0).cpu().float().numpy() 
    image = (image * 255).astype(np.uint8)
    input_image = cv2.circle(image.copy(), (width_circle, height_circle), 15, (255, 255, 255), -1)        
    input_image = cv2.circle(input_image, (width_circle, height_circle), 10, (0, 0, 255), -1)    
            
    input_image = Image.fromarray((input_image).astype(np.uint8))

    return input_image


class QueryKeyVisualizer:
    def __init__(
        self,
        save_timestep_idxs,
        save_layers,
        output_dir,
        model="cogvideox"
    ):
        self.timestep_idxs = save_timestep_idxs
        self.layers = save_layers
        self.output_dir = output_dir
        self.model = model

    def set_attributes(
        self,
        latent_h,
        latent_w,
        latent_f,
        text_len,
        timesteps,
        frames,
    ):
        self.H = latent_h
        self.W = latent_w
        self.latent_num = latent_f
        self.text_len = text_len
        self.background = frames
        self.timesteps = timesteps


    def _get_i2i_attn_map(self, queries, keys, pos, latent_idx, mode='qk'):
        if mode == 'feature':
            queries = queries.unsqueeze(0)
            keys = keys.unsqueeze(0)

        q_start = self.text_len
        k_start = self.text_len + latent_idx * self.H * self.W
        if self.model == "hunyuan_t2v" or mode == 'feature':
            q_start -= self.text_len
            k_start -= self.text_len
        q_end = q_start + self.H * self.W
        k_end = k_start + self.H * self.W
        
        query = queries[:, q_start:q_end]
        key = keys[:, k_start:k_end]

        # Apply softmax to get attention probabilities
        d_k = query.shape[-1]
        attn_score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))        
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
        attn_score = attn_score.mean(0).cpu()

        background = self.background[4 * latent_idx]
        background = background.permute(1, 2, 0).cpu().numpy()
        background = (background * 255).astype(np.uint8)

        img_h, img_w, _ = background.shape
        resize = torchvision.transforms.Resize((img_h, img_w))
        
        attn_score = attn_score.reshape(self.H, self.W, self.H, self.W)[pos[0], pos[1]]
        attn_score = resize(attn_score.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0)
        
        normalizer = mpl.colors.Normalize(vmin=attn_score.min(), vmax=attn_score.max())
        mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        colormapped_im = (mapper.to_rgba(attn_score.to(dtype=torch.float32)[:, :, 0])[:, :, :3] * 255).astype(np.uint8)

        attn_map = cv2.addWeighted(background, 0.3, colormapped_im, 0.7, 0)
        
        return attn_map
    
    
    def _get_t2i_attn_map(self, queries, keys, pos, latent_idx):
        q_start = 0
        k_start = self.text_len + latent_idx * self.H * self.W
        if self.model == "hunyuan_t2v":
            q_start -= self.text_len
            k_start -= self.text_len
        q_end = q_start + self.text_len
        k_end = k_start + self.H * self.W
        
        query = queries[:, q_start:q_end] if q_end > 0 else queries[:, q_start:]
        key = keys[:, k_start:k_end]

        # Apply softmax to get attention probabilities
        d_k = query.shape[-1]
        attn_score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))        
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
        attn_score = attn_score.mean(0).cpu()

        background = self.background[4 * latent_idx]
        background = background.permute(1, 2, 0).cpu().numpy()
        background = (background * 255).astype(np.uint8)

        img_h, img_w, _ = background.shape
        resize = torchvision.transforms.Resize((img_h, img_w))
        
        attn_score = attn_score.reshape(self.text_len, self.H, self.W)[pos]
        attn_score = attn_score.mean(0)
        attn_score = resize(attn_score.unsqueeze(0)).permute(1,2,0)
        
        normalizer = mpl.colors.Normalize(vmin=attn_score.min(), vmax=attn_score.max())
        mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        colormapped_im = (mapper.to_rgba(attn_score.to(dtype=torch.float32)[:, :, 0])[:, :, :3] * 255).astype(np.uint8)

        attn_map = cv2.addWeighted(background, 0.3, colormapped_im, 0.7, 0)
        
        return attn_map
    

    def _pca(self, features):
        combined_features = features.to(torch.float32).cpu().numpy()
        pca = PCA(n_components=3)
        combined_features_pca = pca.fit_transform(combined_features)

        n_features = self.latent_num
        points_per_feature = self.H * self.W
        
        pca_maps = []
        for i in range(n_features):
            start = i * points_per_feature
            end = (i + 1) * points_per_feature

            feature_pca = combined_features_pca[start:end]
            feature_img = feature_pca.reshape(self.H, self.W, 3)

            feature_img_norm = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
            feature_img_norm = (feature_img_norm * 255).astype(np.uint8)
            pca_maps.append(feature_img_norm)

        return pca_maps
        
    
    def save_i2i_attn_map(
        self,
        attn_query_keys,
        pos,
        mode='qk'
    ):
        attnmaps_total = []
        for l, layer in enumerate(self.layers):
            attn_map_per_layer = []
            log_timesteps = []
            for t, t_idx in enumerate(self.timestep_idxs):
                if mode == 'feature':
                    queries = keys = attn_query_keys[t]
                else:
                    queries, keys = attn_query_keys[t]

                attn_map_per_timestep = [
                    self._get_i2i_attn_map(queries[l], keys[l], pos, latent_idx, mode=mode)
                    for latent_idx in range(self.latent_num)
                ]
                attn_map_per_layer.append(attn_map_per_timestep)
                log_timesteps.append(f"timestep{t_idx}({self.timesteps[t_idx]})")
            attnmaps_total.append(attn_map_per_layer)
        return attnmaps_total
    

    def save_pcas(
        self,
        attn_query_keys
    ):
        q_pca_total, k_pca_total = [], []
        for l, layer in enumerate(self.layers):
            query_pca_per_layer = []
            key_pca_per_layer = []
            for t, t_idx in enumerate(self.timestep_idxs):
                queries, keys = attn_query_keys[t]

                if self.model == "hunyuan_t2v":
                    query_pca_list = self._pca(queries[l][:, :-self.text_len].mean(0))
                    key_pca_list = self._pca(keys[l][:, :-self.text_len].mean(0))
                else:
                    query_pca_list = self._pca(queries[l][:, self.text_len:].mean(0))
                    key_pca_list = self._pca(keys[l][:, self.text_len:].mean(0))
                
                query_pca_per_layer.append(query_pca_list)
                key_pca_per_layer.append(key_pca_list)
            q_pca_total.append(query_pca_per_layer)
            k_pca_total.append(key_pca_per_layer)
               
        return q_pca_total, k_pca_total
