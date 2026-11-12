"""
@author: Jingwen Xue
@date: 2026-08-11

@description:
Apply the trained LSTM-EDNet model to synthetic seismic data
for Xline inference, performance evaluation, and visualization.
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
threshold = 0.5


# Load the Trained Model
if not model_path.is_file():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = tf.keras.models.load_model(model_path, compile=False)

print("LSTM-EDNet model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# Load Synthetic Seismic Data
def load_xline_data(data_dir):

    seismic_path = data_dir / "X_test.npy"
    label_path = data_dir / "Y_test.npy"

    if not seismic_path.is_file():
        raise FileNotFoundError(f"Seismic data not found: {seismic_path}")

    if not label_path.is_file():
        raise FileNotFoundError(f"Label data not found: {label_path}")

    xline = np.load(seismic_path)
    label = np.load(label_path)

    if xline.ndim != 2 or label.ndim != 2:
        raise ValueError("Seismic data and labels must be 2D arrays.")

    if xline.shape != label.shape:
        raise ValueError(f"Data and label shapes do not match: "
            f"{xline.shape} vs {label.shape}")

    if xline.shape[0] == 512 and xline.shape[1] != 512:
        pass
    elif xline.shape[1] == 512:
        xline = xline.T
        label = label.T
    else:
        raise ValueError(f"Expected 512 time samples, but received {xline.shape}.")

    xline = xline.astype(np.float32)
    label = (label > 0.5).astype(np.uint8)

    return xline, label


xline, label = load_xline_data(data_dir)

print("\nSynthetic seismic data loaded successfully.")
print("Xline shape:", xline.shape)
print("Label shape:", label.shape)


# Prepare Model Input
X = np.expand_dims(xline.T, axis=-1)

print("Model input shape:", X.shape)

if model.input_shape[1:] != X.shape[1:]:
    raise ValueError(f"Model input shape {model.input_shape[1:]} "
        f"does not match data shape {X.shape[1:]}.")


# Generate Prediction Results
print("\nGenerating Xline predictions...")

prediction_prob = model.predict(X, batch_size=32,verbose=1)

prediction_prob = np.squeeze(prediction_prob, axis=-1)

prediction_binary = (prediction_prob >= threshold).astype(np.uint8)

prediction_prob_show = prediction_prob.T
prediction_binary_show = prediction_binary.T

print("Prediction probability shape:", prediction_prob_show.shape)
print("Binary prediction shape:", prediction_binary_show.shape)


# Calculate Evaluation Metrics
def calculate_binary_metrics(y_true, y_prob, threshold=0.5):

    y_true = np.asarray(y_true).reshape(-1).astype(np.uint8)
    y_prob = np.asarray(y_prob).reshape(-1).astype(np.float32)

    y_pred = (y_prob >= threshold).astype(np.uint8)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    eps = 1e-8

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, y_prob)
    auc = auc_metric.result().numpy()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "F1": f1,
        "IoU": iou,
        "Dice": dice,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)}


metrics = calculate_binary_metrics(label,prediction_prob_show,threshold)


# Print Evaluation Results
print("\n" + "=" * 60)
print("Xline Evaluation Metrics")
print("=" * 60)

for name in [
    "Accuracy",
    "Precision",
    "Recall",
    "AUC",
    "F1",
    "IoU",
    "Dice"]:
    print(f"{name:<12}: {metrics[name]:.4f}")

print("-" * 60)
print(
    f"TP={metrics['TP']}, "
    f"TN={metrics['TN']}, "
    f"FP={metrics['FP']}, "
    f"FN={metrics['FN']}")
print("=" * 60)


# Save Prediction Results
prob_path = data_dir / "synthetic_xline_prediction_probability.npy"
binary_path = data_dir / "synthetic_xline_prediction_binary.npy"

np.save(prob_path, prediction_prob_show)
np.save(binary_path, prediction_binary_show)

print("\nPrediction results saved successfully:")
print(prob_path)
print(binary_path)


# Visualize Xline Prediction Results
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].imshow(xline, cmap="gray", aspect="auto")
axes[0, 0].set_title("Original Seismic")
axes[0, 0].set_xlabel("Trace")
axes[0, 0].set_ylabel("Time Sample")

axes[0, 1].imshow(xline, cmap="gray", aspect="auto")
axes[0, 1].imshow(label,cmap="Reds",alpha=0.4,aspect="auto")
axes[0, 1].set_title("Ground Truth")
axes[0, 1].set_xlabel("Trace")
axes[0, 1].set_ylabel("Time Sample")

im_prob = axes[1, 0].imshow(prediction_prob_show,cmap="jet",
    aspect="auto", vmin=0,vmax=1)
axes[1, 0].set_title("Prediction Probability")
axes[1, 0].set_xlabel("Trace")
axes[1, 0].set_ylabel("Time Sample")
fig.colorbar(im_prob, ax=axes[1, 0], fraction=0.046, pad=0.04)

axes[1, 1].imshow(xline, cmap="gray", aspect="auto")
axes[1, 1].imshow(prediction_binary_show,cmap="Reds",alpha=0.45,aspect="auto")
axes[1, 1].set_title("Binary Prediction (Threshold = 0.5)")
axes[1, 1].set_xlabel("Trace")
axes[1, 1].set_ylabel("Time Sample")

plt.tight_layout()
plt.show()

print("\nXline prediction, evaluation, and visualization completed.")
