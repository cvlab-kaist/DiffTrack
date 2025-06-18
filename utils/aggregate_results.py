import numpy as np
import pandas as pd
import re
from utils.confidence_attention_score import style_top_two

def parse_instance_values(file_path):
    """
    Parse one file and return a 2D list of values.
    Each row corresponds to a layer, each column to a timestep/frame.
    """
    layer_values = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith("Layer"):
                continue
            # Extract just the float numbers
            values = [float(val.strip()) for val in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            layer_values.append(values)
    return np.array(layer_values)  # shape: [num_layers, num_timesteps]


def save_accuracy_mean(file_list, output_path):
    """
    Aggregate mean column-wise (frame-wise) values across multiple files.
    """
    all_arrays = []
    for file_path in file_list:
        arr = parse_instance_values(file_path)
        all_arrays.append(arr)
    
    # Stack and compute mean
    stacked = np.stack(all_arrays, axis=0)  # shape: [num_files, num_layers, num_timesteps]
    mean_values = stacked.mean(axis=0)      # shape: [num_layers, num_timesteps]
    np.savetxt(output_path, mean_values, delimiter=",", fmt="%.4f")

    print(f"Averaged matching accuracy saved to: {output_path}")



def save_score_mean(file_list, output_path):

    # Load all files into list of dataframes
    dataframes = []
    for file in file_list:
        df = pd.read_excel(file, header=None)
        header = df.iloc[0]
        body = df[1:].reset_index(drop=True)
        # Convert numeric columns
        for col in body.columns:
            if col not in [0, 1]:  # skip 'timestep' and 'type'
                body[col] = pd.to_numeric(body[col], errors='coerce')
        dataframes.append(body)

    # Ensure all have the same shape
    shapes = set(df.shape for df in dataframes)
    if len(shapes) != 1:
        raise ValueError("Excel files must all have the same shape")

    # Average the numeric parts
    sum_df = dataframes[0].copy()
    for df_next in dataframes[1:]:
        for col in sum_df.columns:
            if col not in [0, 1]:
                sum_df[col] = sum_df[col] + df_next[col]

    mean_df = sum_df.copy()
    for col in mean_df.columns:
        if col not in [0, 1]:
            mean_df[col] = sum_df[col] / len(dataframes)

    # Reattach the header
    final_df = pd.concat([header.to_frame().T, mean_df], ignore_index=True)

    # Save the result
    final_df.to_excel(output_path, index=False, header=False)

    styled_body = final_df.iloc[1:].reset_index(drop=True)
    styled = styled_body.style
    for col in styled_body.columns:
        if col not in [0, 1]:  # Skip 'timestep' and 'type'
            styled = styled.apply(style_top_two, subset=[col])
    styled.to_excel(output_path, index=False, header=False)

    print(f"Averaged score saved to: {output_path}")
