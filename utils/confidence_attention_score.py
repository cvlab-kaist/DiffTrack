import pandas as pd
import torch
import os

def style_top_two(s):
    """
    s is a Series representing one column (one layer) for all timesteps.
    Because our index is sorted with 3 rows per timestep,
    we process groups of 3 (one for each of self, cross, text).
    """
    styled = [''] * len(s)
    n_groups = len(s) // 3
    for i in range(n_groups):
        group = s.iloc[i*3:(i+1)*3]

        # Convert to numeric, coerce errors to NaN
        group_numeric = pd.to_numeric(group, errors='coerce')

        if group_numeric.isnull().all():
            # Skip styling if all values are invalid
            continue

        top_two = group_numeric.nlargest(2)
        for j, val in enumerate(group_numeric):
            idx = i*3 + j
            if pd.isna(val):
                styled[idx] = ''
            elif val == top_two.iloc[0]:
                styled[idx] = 'font-weight: bold; color: red;'
            elif len(top_two) > 1 and val == top_two.iloc[1]:
                styled[idx] = 'color: orange;'
            else:
                styled[idx] = ''
    return styled

class ConfidenceAttentionScore():
    def __init__(
        self,
        model='cogvideox',
        num_inference_steps=50, 
        num_layers=30,
        mode='max',
        visibility=None,
        text_len=226,
        latent_f=13,
        latent_h=30,
        latent_w=45,
    ):
        time_steps = [f"timestep{j}" for j in range(num_inference_steps)]
        sub_rows = ["self", "cross", "text"]
        index = pd.MultiIndex.from_product([time_steps, sub_rows],
                                        names=["timestep", "type"])
        self.columns = [f"layer{i}" for i in range(num_layers)]
        self.attention_max_df = pd.DataFrame(index=index, columns=self.columns, dtype=float)
        self.attention_sum_df = pd.DataFrame(index=index, columns=self.columns, dtype=float)


        self.model = model
        self.mode = mode
        self.visibility = visibility

        self.text_len = text_len
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_f = latent_f


    def reset(self, text_len, latent_f, latent_h, latent_w):
        self.text_len = text_len
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_f = latent_f
        

    def update(self, attn_weight, layer, timestep_idx):
        text_score, self_score, cross_score = self._update_max(attn_weight)

        text_score_grid_mean = text_score.mean(dim=-1)
        self_score_grid_mean = self_score.mean(dim=-1)
        cross_score_grid_mean = cross_score.mean(dim=-1)

        self.attention_max_df.loc[(f"timestep{timestep_idx}", "self"), f"layer{layer}"] = round(self_score_grid_mean.item(), 4)
        self.attention_max_df.loc[(f"timestep{timestep_idx}", "cross"), f"layer{layer}"] = round(cross_score_grid_mean.item(), 4)
        self.attention_max_df.loc[(f"timestep{timestep_idx}", "text"), f"layer{layer}"] = round(text_score_grid_mean.item(), 4)


        text_score, self_score, cross_score = self._update_sum(attn_weight)

        text_score_grid_mean = text_score.mean(dim=-1)
        self_score_grid_mean = self_score.mean(dim=-1)
        cross_score_grid_mean = cross_score.mean(dim=-1)

        self.attention_sum_df.loc[(f"timestep{timestep_idx}", "self"), f"layer{layer}"] = round(self_score_grid_mean.item(), 4)
        self.attention_sum_df.loc[(f"timestep{timestep_idx}", "cross"), f"layer{layer}"] = round(cross_score_grid_mean.item(), 4)
        self.attention_sum_df.loc[(f"timestep{timestep_idx}", "text"), f"layer{layer}"] = round(text_score_grid_mean.item(), 4)


    def report(self, log_dir='output'):
        styled_max_df = self.attention_max_df.style.apply(style_top_two, subset=self.columns, axis=0)
        styled_max_df.to_excel(os.path.join(log_dir, 'confidence_score.xlsx'), engine='openpyxl')

        styled_sum_df = self.attention_sum_df.style.apply(style_top_two, subset=self.columns, axis=0)
        styled_sum_df.to_excel(os.path.join(log_dir, 'attention_score.xlsx'), engine='openpyxl')
        
        

                
    def _update_max(self, attn_weight):
        f_num, h, w = self.latent_f, self.latent_h, self.latent_w

        if self.model == 'hunyuan_t2v':
            text_score, _ = attn_weight[:, -self.text_len:].max(dim=-1)
            self_score, _ = attn_weight[:, :h * w].max(dim=-1)
        else:
            text_score, _ = attn_weight[:, :self.text_len].max(dim=-1)
            self_score, _ = attn_weight[:, self.text_len: self.text_len + h * w].max(dim=-1)

        cross_score_max_sum = 0
        latent_visible_mask_sum = 0
        for f in range(1, f_num):
            if self.model == 'hunyuan_t2v':
                cross_score_max, _ = attn_weight[:, f * h * w:(f + 1) * h * w].max(dim=-1)
            else:
                cross_score_max, _ = attn_weight[:, self.text_len + f * h * w: self.text_len + (f + 1) * h * w].max(dim=-1)

            latent_visible_mask = self.visibility[0][4*f-3:4*f+1].to(cross_score_max.device).all(dim=0).float()
            cross_score_max = cross_score_max * latent_visible_mask
            
            cross_score_max_sum += cross_score_max
            latent_visible_mask_sum += latent_visible_mask
        cross_score = cross_score_max_sum / torch.clip(latent_visible_mask_sum, 1e-8)

        return text_score, self_score, cross_score
    

    def _update_sum(self, attn_weight):
        h, w = self.latent_h, self.latent_w

        attn_weight = attn_weight[self.visibility[0,0]]
        
        if self.model == 'hunyuan_t2v':
            text_score = attn_weight[:, -self.text_len:].sum(dim=-1)
            self_score = attn_weight[:, :h * w].sum(dim=-1)
            cross_score = attn_weight[:, h * w:-self.text_len].sum(dim=-1)
        else:
            text_score = attn_weight[:, :self.text_len].sum(dim=-1)
            self_score = attn_weight[:, self.text_len: self.text_len + h * w].sum(dim=-1)
            cross_score = attn_weight[:, self.text_len + h * w:].sum(dim=-1)
        
        return text_score, self_score, cross_score