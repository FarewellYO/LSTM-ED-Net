"""
@author: Jingwen Xue
@date: 2026-06-04
@description:
Load the trained LSTM-EDNet model, evaluate its performance on the
training, validation, and test datasets, and visualize prediction
results on the test dataset.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Path Configuration
if "__file__" in globals():
    data_dir = Path(__file__).resolve().parent
else:
    data_dir = Path.cwd()

model_path = data_dir / "lstm_ednet_model.keras"


# Load Seismic Data
def load_seismic_data(data_dir):

    def _load_one_pair(x_file, y_file):
        x_path = data_dir / x_file
        y_path = data_dir / y_file

        if not x_path.is_file():
            raise FileNotFoundError(f"Data file not found: {x_path}")
        if not y_path.is_file():
            raise FileNotFoundError(f"Label file not found: {y_path}")

        X = np.load(x_path)
        Y = np.load(y_path)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("Seismic data and labels must be 2D arrays.")

        if X.shape != Y.shape:
            raise ValueError(
                f"Data and label shapes do not match: {X.shape} vs {Y.shape}")

        # Convert to (number of traces, 512 samples)
        if X.shape[0] == 512 and X.shape[1] != 512:
            X = X.T
            Y = Y.T
        elif X.shape[1] == 512:
            pass
        else:
            raise ValueError(
                f"Expected 512 time samples, but received {X.shape}.")

        X = np.expand_dims(X, axis=-1).astype(np.float32)
        Y = np.expand_dims((Y > 0.5).astype(np.float32), axis=-1)

        return X, Y

    X_train, Y_train = _load_one_pair("X_train.npy", "Y_train.npy")
    X_val, Y_val = _load_one_pair("X_val.npy", "Y_val.npy")
    X_test, Y_test = _load_one_pair("X_test.npy", "Y_test.npy")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# Load Dataset and Trained Model
print("Loading data...")
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_seismic_data(data_dir)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

if not model_path.is_file():
    raise FileNotFoundError(f"Trained model not found: {model_path}")

print("Loading the saved model...")
salt_model = load_model(model_path, compile=False)
print("Model loaded successfully.")


# Calculate Evaluation Metrics
def calculate_binary_metrics(model, X, Y, threshold=0.5, batch_size=32):
    Y_prob = model.predict(X, batch_size=batch_size, verbose=0)

    y_true = (Y.reshape(-1) > 0.5).astype(np.int32)
    y_prob = Y_prob.reshape(-1).astype(np.float32)

    y_true_float = y_true.astype(np.float32)
    y_prob_clip = np.clip(y_prob, 1e-8, 1.0 - 1e-8)

    loss = -np.mean(y_true_float * np.log(y_prob_clip)
        + (1.0 - y_true_float) * np.log(1.0 - y_prob_clip))

    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    eps = 1e-8

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, y_prob)
    auc = auc_metric.result().numpy()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "f1_score": f1_score,
        "iou": iou,
        "dice": dice,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)}


def print_metrics_table(train_metrics, val_metrics, test_metrics):
    metric_names = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "auc",
        "f1_score",
        "iou",
        "dice"]

    print("\n" + "=" * 80)
    print("Detailed Evaluation Metrics for Training, Validation, and Test Sets")
    print("=" * 80)

    for name in metric_names:
        print(
            f"{name.upper():<12} "
            f"Training: {train_metrics[name]:.4f}  "
            f"Validation: {val_metrics[name]:.4f}  "
            f"Test: {test_metrics[name]:.4f}")

    print("-" * 80)
    print("Test Set Confusion Matrix Statistics")
    print(
        f"TP: {test_metrics['tp']}  "
        f"TN: {test_metrics['tn']}  "
        f"FP: {test_metrics['fp']}  "
        f"FN: {test_metrics['fn']}")
    print("=" * 80)


# Calculate Metrics for Training, Validation, and Test Sets
train_metrics = calculate_binary_metrics(salt_model, X_train, Y_train, threshold=0.5)
val_metrics = calculate_binary_metrics(salt_model, X_val, Y_val, threshold=0.5)
test_metrics = calculate_binary_metrics(salt_model, X_test, Y_test, threshold=0.5)

print_metrics_table(train_metrics, val_metrics, test_metrics)


# Visualize Prediction Results on the Test Dataset
def visualize_test_results(X_test, Y_test, model):
    Y_pred_prob = model.predict(X_test, batch_size=32, verbose=0)
    Y_pred_binary = (Y_pred_prob > 0.5).astype(np.float32)

    print("Generating prediction visualization for the test dataset...")

    seismic_img = X_test[:, :, 0].T
    gt_mask = Y_test[:, :, 0].T
    prob_map = Y_pred_prob[:, :, 0].T
    binary_mask = Y_pred_binary[:, :, 0].T

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(seismic_img, cmap="gray", aspect="auto")
    plt.title("Original Seismic")
    plt.xlabel("Trace")
    plt.ylabel("Time Sample")

    plt.subplot(2, 2, 2)
    plt.imshow(seismic_img, cmap="gray", aspect="auto")
    plt.imshow(gt_mask, cmap="Reds", alpha=0.4, aspect="auto")
    plt.title("Ground Truth")

    plt.subplot(2, 2, 3)
    plt.imshow(seismic_img, cmap="gray", aspect="auto")
    im_prob = plt.imshow(
        prob_map,
        cmap="jet",
        alpha=0.5,
        aspect="auto",
        vmin=0,
        vmax=1)
    plt.title("Prediction Probability")
    plt.colorbar(im_prob, fraction=0.046, pad=0.04)

    plt.subplot(2, 2, 4)
    plt.imshow(seismic_img, cmap="gray", aspect="auto")
    plt.imshow(binary_mask, cmap="Blues", alpha=0.5, aspect="auto")
    plt.title("Binary Prediction (threshold = 0.5)")

    plt.tight_layout()
    plt.show()


visualize_test_results(X_test, Y_test, salt_model)

print("\nModel evaluation and visualization completed.")