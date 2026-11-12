"""
@author: Jingwen Xue
@date: 2026-08-08
@description:
Apply the trained LSTM-EDNet model to synthetic seismic Inline data,
generate prediction probability maps and binary predictions,
and visualize the results.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Path Configuration
if "__file__" in globals():
    data_dir = Path(__file__).resolve().parent
else:
    data_dir = Path.cwd()

model_path = data_dir / "lstm_ednet_model.keras"


# Load the Trained LSTM-EDNet Model
if not model_path.is_file():
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model = tf.keras.models.load_model(model_path, compile=False)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# Load Synthetic Seismic Data
def load_inline_data(data_dir):

    seismic_path = data_dir / "X_test.npy"
    label_path = data_dir / "Y_test.npy"

    if not seismic_path.is_file():
        raise FileNotFoundError(f"Seismic data not found: {seismic_path}")

    if not label_path.is_file():
        raise FileNotFoundError(f"Label data not found: {label_path}")

    inline_data = np.load(seismic_path)
    inline_label = np.load(label_path)

    if inline_data.ndim != 2 or inline_label.ndim != 2:
        raise ValueError("Seismic data and labels must be 2D arrays.")

    if inline_data.shape != inline_label.shape:
        raise ValueError(
            f"Data and label shapes do not match: "
            f"{inline_data.shape} vs {inline_label.shape}")

    # Convert data to (number of traces, 512 samples)
    if inline_data.shape[0] == 512 and inline_data.shape[1] != 512:
        inline_data = inline_data.T
        inline_label = inline_label.T
    elif inline_data.shape[1] == 512:
        pass
    else:
        raise ValueError(
            f"Expected 512 time samples, but received {inline_data.shape}.")

    inline_data = inline_data.astype(np.float32)
    inline_label = (inline_label > 0.5).astype(np.uint8)

    return inline_data, inline_label


inline_norm, inline_label = load_inline_data(data_dir)

print("\nSynthetic Inline data loaded successfully.")
print("Inline seismic shape:", inline_norm.shape)
print("Inline label shape:", inline_label.shape)


# Prepare Model Input
inline_input = inline_norm

X = np.expand_dims(inline_input, axis=-1)

print("Model input shape:", X.shape)

if model.input_shape[1:] != X.shape[1:]:
    raise ValueError(
        f"Model input shape {model.input_shape[1:]} "
        f"does not match data shape {X.shape[1:]}.")


# Generate Prediction Results
print("\nGenerating Inline predictions...")

prediction_prob = model.predict(X, batch_size=32, verbose=1)

prediction_prob = np.squeeze(prediction_prob, axis=-1)

threshold = 0.5

prediction_binary = (prediction_prob >= threshold).astype(np.uint8)

print("Prediction completed.")
print("Prediction probability shape:", prediction_prob.shape)
print("Binary prediction shape:", prediction_binary.shape)


# Save Prediction Results
prediction_prob_save = prediction_prob.T
prediction_binary_save = prediction_binary.T

np.save(data_dir / "inline_test_prediction_probability.npy", prediction_prob_save)

np.save( data_dir / "inline_test_prediction_binary.npy", prediction_binary_save)

print("\nPrediction results saved successfully.")
print("Probability:", data_dir / "inline_test_prediction_probability.npy")
print("Binary:", data_dir / "inline_test_prediction_binary.npy")


# Visualize Prediction Results
seismic_img = inline_norm.T
gt_mask = inline_label.T
prob_map = prediction_prob.T
binary_mask = prediction_binary.T

plt.figure(figsize=(14, 10))

# Original Seismic
plt.subplot(2, 2, 1)
plt.imshow(seismic_img, cmap="gray", aspect="auto")
plt.title("Original Seismic")
plt.xlabel("Trace")
plt.ylabel("Time Sample")

# Ground Truth
plt.subplot(2, 2, 2)
plt.imshow(seismic_img, cmap="gray", aspect="auto")
plt.imshow(gt_mask,cmap="Reds",
    alpha=0.4, aspect="auto")
plt.title("Ground Truth")
plt.xlabel("Trace")
plt.ylabel("Time Sample")

# Prediction Probability
plt.subplot(2, 2, 3)
plt.imshow(seismic_img, cmap="gray", aspect="auto")
im_prob = plt.imshow(prob_map, cmap="jet", alpha=0.5,aspect="auto", vmin=0, vmax=1)
plt.title("Prediction Probability")
plt.xlabel("Trace")
plt.ylabel("Time Sample")
plt.colorbar(im_prob, fraction=0.046, pad=0.04)

# Binary Prediction
plt.subplot(2, 2, 4)
plt.imshow(seismic_img, cmap="gray", aspect="auto")
plt.imshow(binary_mask, cmap="Reds", alpha=0.45, aspect="auto")
plt.title("Binary Prediction (Threshold = 0.5)")
plt.xlabel("Trace")
plt.ylabel("Time Sample")

plt.tight_layout()
plt.show()

print("\nInline prediction and visualization completed.")
