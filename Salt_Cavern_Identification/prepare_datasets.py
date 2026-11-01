# -*- coding: utf-8 -*-
"""
@author: Jingwen Xue
@date: 2026-06-03
@description:
Prepare datasets for LSTM-ED-Net-based salt-cavern seismic identification.
"""

#%%
import os
import numpy as np
import pandas as pd

#%% Path Configuration
csv_path = r"The_path_to_metadata_file"
data_root =r"The_path_to_seismic_data"

label_root = r"The_path_to_label_data"
save_dir =r"The_path_to_output_directory"
os.makedirs(save_dir, exist_ok=True)

#%% Training, Validation, and Test Set Allocation
df = pd.read_csv(csv_path)

test_inlines_pos = [] # Inline profiles reserved for testin
val_inlines_pos = [] # Inline profiles reserved for validation

bg_train_inlines = [] # Profiles without salt cavern for training
bg_val_inlines = [] # Profiles without salt cavern for validation
bg_test_inlines = [] # Profiles without salt cavern for testing

all_salt_inlines = df[df['salt_name'].str.contains(sonar_name, na=False)]['inline_no'].unique()

train_pos_inlines = [
    i for i in all_salt_inlines
    if i not in test_inlines_pos + val_inlines_pos
]

#%% Data Extraction Function
def extract_and_prepare(inline_list, mode_name):
   
    inline_start = None  # Starting Inline number
    x_list, y_list = [], []
    print(f"Processing {mode_name} dataset, containing {len(inline_list)} Inline profiles...")

    for inline_no in inline_list:
        rel_idx = inline_no - inline_start

        feature_file = os.path.join(data_root, f"inline_{rel_idx:03d}.npy")
        mask_file = os.path.join(label_root, f"inline_{rel_idx:03d}_label.npy")

        if not os.path.exists(feature_file):
            print(f"Skipping: data file not found {feature_file}")
            continue

        data_mat = np.load(feature_file)

        if os.path.exists(mask_file):
            mask_mat = np.load(mask_file)
        else:
            mask_mat = np.zeros_like(data_mat)

        if data_mat.shape != mask_mat.shape:
            raise ValueError(
                f"Data and label shapes do not match for Inline {inline_no}:"
                f"data_mat.shape={data_mat.shape}, mask_mat.shape={mask_mat.shape}"
            )

        num_traces = data_mat.shape[1]

        for j in range(num_traces):
            x_list.append(data_mat[:, j])
            y_list.append(mask_mat[:, j])

    final_x = np.array(x_list).astype(np.float32)
    final_y = np.array(y_list).astype(np.float32)

    np.save(os.path.join(save_dir, f"{mode_name}_x.npy"), final_x)
    np.save(os.path.join(save_dir, f"{mode_name}_y.npy"), final_y)

    print(f"{mode_name} dataset extraction completed! Shape: {final_x.shape}")

#%% Positive and Negative Sample Statistics for Each Dataset
def summarize_saved_labels():
    
    stats_list = []

    for mode_name in ["train", "validation", "test"]:
        y_path = os.path.join(save_dir, f"{mode_name}_y.npy")

        if not os.path.exists(y_path):
            print(f"{mode_name}_y.npy not found. Skipping statistics.")
            continue

        y_data = np.load(y_path)

        total_points = int(y_data.size)
        positive_points = int(np.sum(y_data > 0.5))
        negative_points = total_points - positive_points

        positive_ratio = positive_points / total_points if total_points > 0 else 0
        negative_ratio = negative_points / total_points if total_points > 0 else 0

        if positive_points > 0:
            neg_pos_ratio = negative_points / positive_points
        else:
            neg_pos_ratio = np.inf

        print("-" * 80)
        print(f"{mode_name} Sample-Level Label Statistics")
        print("-" * 80)
        print(f"Total number of samples: {total_points}")
        print(f"Number of positive samples (label=1): {positive_points}")
        print(f"Number of negative samples (label=0): {negative_points}")
        print(f"Positive sample ratio: {positive_ratio:.6f}")
        print(f"Negative sample ratio: {negative_ratio:.6f}")
        print(f"Negative-to-positive sample ratio: {neg_pos_ratio:.2f}")
        print("-" * 80)

        stats_list.append({
            "dataset": mode_name,
            "total_points": total_points,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "negative_positive_ratio": neg_pos_ratio
        })

    stats_df = pd.DataFrame(stats_list)
    stats_save_path = os.path.join(save_dir, "label_point_statistics.csv")
    stats_df.to_csv(stats_save_path, index=False, encoding="utf-8-sig")

    print(f"Label statistics saved to: {stats_save_path}")

#%% Data Normalization
def global_normalize_all():
    train_x_path = os.path.join(save_dir, "train_x.npy")
    val_x_path = os.path.join(save_dir, "validation_x.npy")
    test_x_path = os.path.join(save_dir, "test_x.npy")

    if not os.path.exists(train_x_path):
        print("Training dataset not found. Unable to calculate global statistics.")
        return

    print("Calculating global statistics from the training dataset...")
    train_x = np.load(train_x_path)

    global_mean = np.mean(train_x)
    global_std = np.std(train_x)

    print(f">>> Training dataset global mean: {global_mean:.6f}")
    print(f">>> Training dataset global standard deviation: {global_std:.6f}")

    mean_save_path = os.path.join(save_dir, "global_mean.npy")
    std_save_path = os.path.join(save_dir, "global_std.npy")
    np.save(mean_save_path, global_mean)
    np.save(std_save_path, global_std)
    print("Mean save path:", mean_save_path)
    print("Standard deviation save path:", std_save_path)
    print("Mean file exists:", os.path.exists(mean_save_path))
    print("Standard deviation file exists:", os.path.exists(std_save_path))

    def apply_norm(path, mode_name):
        if os.path.exists(path):
            data = np.load(path)
            normed_data = (data - global_mean) / (global_std + 1e-8)

            save_path = os.path.join(save_dir, f"{mode_name}_x_n.npy")
            np.save(save_path, normed_data.astype(np.float32))

            print(f"Global normalization completed for {mode_name} -> {mode_name}_x_n.npy")
        else:
            print(f"⚠ Skipping {mode_name}: file not found.")

    apply_norm(train_x_path, "train")
    apply_norm(val_x_path, "validation")
    apply_norm(test_x_path, "test")

#%% Extract Training, Validation, and Test Datasets
extract_and_prepare(train_pos_inlines + bg_train_inlines, "train")
extract_and_prepare(val_inlines_pos + bg_val_inlines, "validation")
extract_and_prepare(test_inlines_pos + bg_test_inlines, "test")

#%% Apply Normalization
global_normalize_all()

#%% Normalization of All Inline Profiles
def normalize_all_inline():
    inline_dir = data_root
    inline_norm_dir = os.path.join(os.path.dirname(inline_dir), "inline_slices_normalized")
    os.makedirs(inline_norm_dir, exist_ok=True)

    mean_path = os.path.join(save_dir, "global_mean.npy")
    std_path = os.path.join(save_dir, "global_std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("global_mean.npy or global_std.npy not found. Please run global_normalize_all() first.")
        return
    global_mean = np.load(mean_path)
    global_std = np.load(std_path)

    inline_files = sorted([f for f in os.listdir(inline_dir) if f.endswith(".npy")])

    print(f"A total of {len(inline_files)} Inline profiles found.")

   
    for file in inline_files:
        inline_path = os.path.join(inline_dir, file)
        inline_data = np.load(inline_path)

        inline_norm = (inline_data - global_mean) / (global_std + 1e-8)

        save_name = file.replace(".npy", "_n.npy")
        save_path = os.path.join(inline_norm_dir, save_name)
        np.save(save_path, inline_norm.astype(np.float32))
        print(f"{file} -> {save_name}  Shape:{inline_norm.shape}")

    print("Normalization of all Inline profiles completed!")

#%% Normalize All Inline Profiles
normalize_all_inline()