'''
@author: Jingwen Xue
@date: 2026-06-04
@description:
This script loads the trained LSTM-EDNet model, evaluates its performance, and visualizes prediction results on the test dataset.
'''

#%% 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

#%% Path Configuration
data_path = r"The_path_to_dataset_directory"

model_path = r"The_path_to_trained_model.keras"

#%% Load Seismic Data
def load_seismic_data(data_dir):
    def _load_one_pair(x_file, y_file):
        X = np.load(os.path.join(data_dir, x_file))
        Y = np.load(os.path.join(data_dir, y_file))

        if X.shape[1] != 512:
            X = X.T
        if Y.shape[1] != 512:
            Y = Y.T

        X = np.expand_dims(X, axis=-1).astype(np.float32)
        Y = np.expand_dims(Y, axis=-1).astype(np.float32)
        Y = (Y > 0.5).astype(np.int32)

        return X, Y

    X_train, Y_train = _load_one_pair('train_x_n.npy', 'train_y.npy')
    X_val, Y_val = _load_one_pair('validation_x_n.npy', 'validation_y.npy')
    X_test, Y_test = _load_one_pair('test_x_n.npy', 'test_y.npy')

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

#%% Load Dataset and Trained Model
print("Loading data...")
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_seismic_data(data_path)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

print("Loading the saved model...")
salt_model = load_model(model_path, compile=False)

print("Model loaded successfully and is ready for prediction, evaluation, and visualization.")

#%% Calculate Evaluation Metrics
def calculate_binary_metrics(model, X, Y, threshold=0.5, batch_size=32):
    Y_prob = model.predict(X, batch_size=batch_size, verbose=0)

    y_true = (Y.reshape(-1) > 0.5).astype(np.int32)
    y_prob = Y_prob.reshape(-1).astype(np.float32)

    y_true_float = y_true.astype(np.float32)
    y_prob_clip = np.clip(y_prob, 1e-8, 1.0 - 1e-8)

    loss = -np.mean(
        y_true_float * np.log(y_prob_clip) +
        (1.0 - y_true_float) * np.log(1.0 - y_prob_clip)
    )

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


#%% Calculate Metrics for Training, Validation, and Test Sets
train_metrics = calculate_binary_metrics(salt_model, X_train, Y_train, threshold=0.5)
val_metrics = calculate_binary_metrics(salt_model, X_val, Y_val, threshold=0.5)
test_metrics = calculate_binary_metrics(salt_model, X_test, Y_test, threshold=0.5)

print_metrics_table(train_metrics, val_metrics, test_metrics)

#%% Visualize Prediction Results on the Test Dataset
def visualize_test_results(X_test, Y_test, model, n_traces_per_profile=154, max_profiles=6):

    Y_pred_prob = model.predict(X_test, batch_size=32)
    Y_pred_binary = (Y_pred_prob > 0.5).astype(np.float32)

    total_traces = X_test.shape[0]

    if total_traces < n_traces_per_profile:
        n_traces_per_profile = total_traces

    num_profiles = total_traces // n_traces_per_profile
    num_profiles = min(num_profiles, max_profiles)

    print(f"Number of profiles to visualize: {num_profiles}")

    for i in range(num_profiles):
        start = i * n_traces_per_profile
        end = (i + 1) * n_traces_per_profile

        seismic_img = X_test[start:end, :, 0].T
        gt_mask = Y_test[start:end, :, 0].T
        prob_map = Y_pred_prob[start:end, :, 0].T
        binary_mask = Y_pred_binary[start:end, :, 0].T

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(seismic_img, cmap='gray', aspect='auto')
        plt.title(f"Profile {i + 1}: Original Seismic")
        plt.xlabel("Trace")
        plt.ylabel("Time Sample")

        plt.subplot(2, 2, 2)
        plt.imshow(seismic_img, cmap='gray', aspect='auto')
        plt.imshow(gt_mask, cmap='Reds', alpha=0.4, aspect='auto')
        plt.title("Ground Truth")

        plt.subplot(2, 2, 3)
        plt.imshow(seismic_img, cmap='gray', aspect='auto')
        im_prob = plt.imshow(prob_map, cmap='jet', alpha=0.5, aspect='auto', vmin=0, vmax=1)
        plt.title("Prediction Probability")
        plt.colorbar(im_prob, fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 4)
        plt.imshow(seismic_img, cmap='gray', aspect='auto')
        plt.imshow(binary_mask, cmap='Blues', alpha=0.5, aspect='auto')
        plt.title("Binary Prediction, threshold=0.5")

        plt.tight_layout()
        plt.show()


visualize_test_results(X_test, Y_test, salt_model)

