import pandas as pd
import numpy as np

def normalize(df):
    return (df - df.values.min()) / (df.values.max() - df.values.min())

def save_harmonic_mean(acc_path, conf_path, attn_path, output_path):
    acc_df = pd.read_csv(acc_path, header=None).transpose() 
    conf_df = pd.read_excel(conf_path, header=None) 
    attn_df = pd.read_excel(attn_path, header=None)

    acc_df = acc_df.iloc[1:].reset_index(drop=True)
    conf_df = conf_df[conf_df.iloc[:, 1] == 'cross'].iloc[:, 2:].reset_index(drop=True)
    attn_df = attn_df[attn_df.iloc[:, 1] == 'cross'].iloc[:, 2:].reset_index(drop=True)

    acc_df.columns = range(acc_df.shape[1])
    conf_df.columns = range(conf_df.shape[1])
    attn_df.columns = range(attn_df.shape[1])

    acc_values = normalize(acc_df.astype(float).copy())
    conf_values = normalize(conf_df.astype(float).copy())
    attn_values = normalize(attn_df.astype(float).copy())

    denominator = (1 / (acc_values)) + (1 / (conf_values)) + (1 / (attn_values))
    df_harmonic_mean = 3 / denominator

    df_harmonic_mean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_harmonic_mean.fillna(0, inplace=True)

    df_harmonic_mean.to_csv(output_path, index=False, header=False)

    print(f"Harmonic mean saved to: {output_path}")