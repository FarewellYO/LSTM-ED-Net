"""
@author: Jingwen Xue
@date: 2026-06-03
@description:
Split synthetic seismic data into training, validation, and test sets,
perform global normalization, and save the processed datasets for
LSTM-ED-Net training and evaluation.
"""

import numpy as np
from pathlib import Path


# Path Configuration
if "__file__" in globals():
    data_dir = Path(__file__).resolve().parent
else:
    data_dir = Path.cwd()


# Load Synthetic Seismic Data and Labels
seismic_path = data_dir / "SeismicData.npy"
label_path = data_dir / "LabelData.npy"

if not seismic_path.is_file():
    raise FileNotFoundError(f"Seismic data not found: {seismic_path}")

if not label_path.is_file():
    raise FileNotFoundError(f"Label data not found: {label_path}")

SeismicDataNetwork = np.load(seismic_path)
LabelDataNetwork = np.load(label_path)

print("SeismicDataNetwork shape:", SeismicDataNetwork.shape)
print("LabelDataNetwork shape:", LabelDataNetwork.shape)

if SeismicDataNetwork.ndim != 2 or LabelDataNetwork.ndim != 2:
    raise ValueError("Seismic data and label data must be two-dimensional.")

if SeismicDataNetwork.shape != LabelDataNetwork.shape:
    raise ValueError("Seismic data and label data shapes do not match.")

n_samples, n_traces = SeismicDataNetwork.shape


# Split Training, Validation, and Test Datasets
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, \
    "The sum of dataset split ratios must equal 1."

seed = 42
rng = np.random.default_rng(seed)

trace_indices = np.arange(n_traces)
rng.shuffle(trace_indices)

n_train = int(n_traces * train_ratio)
n_val = int(n_traces * val_ratio)
n_test = n_traces - n_train - n_val

train_idx = trace_indices[:n_train]
val_idx = trace_indices[n_train:n_train + n_val]
test_idx = trace_indices[n_train + n_val:]

print("\nDataset allocation:")
print("Training traces:", len(train_idx))
print("Validation traces:", len(val_idx))
print("Test traces:", len(test_idx))

X_train = SeismicDataNetwork[:, train_idx]
Y_train = LabelDataNetwork[:, train_idx]

X_val = SeismicDataNetwork[:, val_idx]
Y_val = LabelDataNetwork[:, val_idx]

X_test = SeismicDataNetwork[:, test_idx]
Y_test = LabelDataNetwork[:, test_idx]

print("\nDataset shapes:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


# Data Normalization
train_mean = X_train.mean()
train_std = X_train.std()

if train_std == 0:
    train_std = 1.0

X_train_norm = (X_train - train_mean) / train_std
X_val_norm = (X_val - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

Y_train = Y_train.astype(np.uint8)
Y_val = Y_val.astype(np.uint8)
Y_test = Y_test.astype(np.uint8)

print("\nNormalization completed.")
print("Training mean:", train_mean)
print("Training standard deviation:", train_std)


# Save Processed Datasets
np.save(data_dir / "X_train.npy", X_train_norm)
np.save(data_dir / "Y_train.npy", Y_train)

np.save(data_dir / "X_val.npy", X_val_norm)
np.save(data_dir / "Y_val.npy", Y_val)

np.save(data_dir / "X_test.npy", X_test_norm)
np.save(data_dir / "Y_test.npy", Y_test)

# Save Normalization Parameters
np.save(data_dir / "train_mean.npy", train_mean)
np.save(data_dir / "train_std.npy", train_std)

print("\nProcessed datasets saved successfully:")
print(data_dir / "X_train.npy")
print(data_dir / "Y_train.npy")
print(data_dir / "X_val.npy")
print(data_dir / "Y_val.npy")
print(data_dir / "X_test.npy")
print(data_dir / "Y_test.npy")
print(data_dir / "train_mean.npy")
print(data_dir / "train_std.npy")

print("\nAll dataset preparation steps completed successfully.")
