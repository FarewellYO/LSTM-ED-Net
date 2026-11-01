"""
@author: Jingwen Xue
@date: 2026-08-11

@description:
This script applies the trained LSTM-EDNet model to Xline seismic profiles for cross-direction prediction, performance evaluation, and full-volume visualization.
"""

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix

#%% Path configuration
model_path = r"The_path_to_trained_model.keras"
xline_norm_dir = r"The_path_to_normalized_xline_data"
xline_label_dir = r"The_path_to_xline_labels"
xline_prob_save_dir = r"The_path_to_probability_results"
xline_binary_save_dir = r"The_path_to_binary_results"
os.makedirs(xline_prob_save_dir, exist_ok=True)
os.makedirs(xline_binary_save_dir, exist_ok=True)

#%% Load the model
model = tf.keras.models.load_model(model_path, compile=False)
print("LSTM-EDNet model loaded successfully.")

#%% Select Xline Profiles for Evaluation
xline_list = sorted([
    file for file in os.listdir(xline_norm_dir)
    if file.endswith("_n.npy")
    and os.path.exists(
        os.path.join(
            xline_label_dir,
            file.replace("_n.npy", "_label.npy")))])

if not xline_list:
    raise FileNotFoundError(
        "No Xline profiles with corresponding labels were found."
    )

print(f"Number of Xline profiles selected for evaluation: {len(xline_list)}")

#%% Evaluation Metrics Function
def calculate_binary_metrics(y_true, y_prob, threshold=0.5):

    y_true = np.asarray(y_true).reshape(-1).astype(np.uint8)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.uint8)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    intersection = tp
    union = tp + fp + fn

    iou = intersection / union if union > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "F1": f1,
        "IoU": iou,
        "Dice": dice,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn}

#%% Predict Xlne Profiles and Calculate Evaluation Metrics
threshold = 0.5
binary_cmap = ListedColormap(["#F2F2F2","#2A9D8F"])

all_true = []
all_prob = []

for xline_file in xline_list:

    print("\nProcessing:", xline_file)

    xline_path = os.path.join(xline_norm_dir, xline_file)
    xline = np.load(xline_path)

    print("Original xline shape:", xline.shape)

    xline_name = xline_file.replace("_n.npy", "")
    label_path = os.path.join(xline_label_dir,f"{xline_name}_label.npy")
    label = np.load(label_path)
    print("Label shape:", label.shape)

    xline_input = xline.T
    X = np.expand_dims(xline_input, axis=-1)
    print("Model input shape:", X.shape)

    prediction_prob = model.predict(X,batch_size=32,verbose=0)
    prediction_prob = np.squeeze(prediction_prob)

    label_model = label.T
    print("Prediction shape:", prediction_prob.shape)
    print("Label model shape:", label_model.shape)

    metrics = calculate_binary_metrics(label_model,prediction_prob,threshold)

    print("\nCurrent Xline Evaluation Metrics:")
    print(f"Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"Precision : {metrics['Precision']:.4f}")
    print(f"Recall    : {metrics['Recall']:.4f}")
    print(f"AUC       : {metrics['AUC']:.4f}")
    print(f"F1        : {metrics['F1']:.4f}")
    print(f"IoU       : {metrics['IoU']:.4f}")
    print(f"Dice      : {metrics['Dice']:.4f}")

    print(
        f"TP={metrics['TP']}, "
        f"TN={metrics['TN']}, "
        f"FP={metrics['FP']}, "
        f"FN={metrics['FN']}")

    all_true.append(label_model.reshape(-1))
    all_prob.append(prediction_prob.reshape(-1))

    prediction_binary = (prediction_prob >= threshold).astype(np.uint8)

    probability_show = prediction_prob.T
    binary_show = prediction_binary.T

    plt.figure(figsize=(16,6))

    plt.subplot(1,3,1)
    plt.imshow(xline,cmap="gray",aspect="auto")
    plt.title(f"{xline_name} Original Xline")
    plt.xlabel("Inline")
    plt.ylabel("Time Sample")

    plt.subplot(1,3,2)
    plt.imshow(probability_show,cmap="viridis",aspect="auto",vmin=0,vmax=1)
    plt.colorbar()
    plt.title("Prediction Probability")
    plt.xlabel("Inline")
    plt.ylabel("Time Sample")

    plt.subplot(1,3,3)
    plt.imshow(binary_show,cmap=binary_cmap,aspect="auto")
    plt.title("Binary Prediction")
    plt.xlabel("Inline")
    plt.ylabel("Time Sample")

    plt.tight_layout()
    plt.show()

#%% Calculate Overall Evaluation Metrics for All Selected Xline Profiles
all_true_total = np.concatenate(all_true)
all_prob_total = np.concatenate(all_prob)

overall_metrics = calculate_binary_metrics(all_true_total,all_prob_total,threshold)

print("\n" + "=" * 40)
print("Overall Evaluation Metrics for All Selected Xline Profiles")
print("=" * 40)

print(f"Accuracy  : {overall_metrics['Accuracy']:.4f}")
print(f"Precision : {overall_metrics['Precision']:.4f}")
print(f"Recall    : {overall_metrics['Recall']:.4f}")
print(f"AUC       : {overall_metrics['AUC']:.4f}")
print(f"F1        : {overall_metrics['F1']:.4f}")
print(f"IoU       : {overall_metrics['IoU']:.4f}")
print(f"Dice      : {overall_metrics['Dice']:.4f}")

print(
    f"TP={overall_metrics['TP']}, "
    f"TN={overall_metrics['TN']}, "
    f"FP={overall_metrics['FP']}, "
    f"FN={overall_metrics['FN']}"
)

#%% Batch Prediction for All Xline Profiles
xline_files = sorted([f for f in os.listdir(xline_norm_dir) if f.endswith("_n.npy")])
print(f"A total of {len(xline_files)} Xline/CDP profiles found.")

threshold = 0.5
for i, file in enumerate(xline_files):
    print(f"Processing {i + 1}/{len(xline_files)}: {file}")

    xline_path = os.path.join(xline_norm_dir, file)
    xline_norm = np.load(xline_path)

    print("Xline shape:", xline_norm.shape)

    xline_input = xline_norm.T

    X = np.expand_dims(xline_input, axis=-1)

    prediction_prob = model.predict(X, batch_size=32, verbose=0)
    prediction_prob = np.squeeze(prediction_prob)

    prediction_binary = (prediction_prob >= threshold).astype(np.uint8)

    prediction_prob_save = prediction_prob.T
    prediction_binary_save = prediction_binary.T

    xline_name = file.replace("_n.npy", "")

    np.save(os.path.join(xline_prob_save_dir, f"{xline_name}_probability.npy"), prediction_prob_save)

    np.save(os.path.join(xline_binary_save_dir, f"{xline_name}_binary.npy"), prediction_binary_save)
    print(f"{xline_name} processing completed. Saved shape: {prediction_prob_save.shape}")

print("Batch prediction for all Xline/CDP profiles completed!")

#%% View an Example of Full-Volume Xline Prediction Results
example_xline_file = xline_files[0].replace("_n.npy", "")

prob_path = os.path.join(xline_prob_save_dir, f"{example_xline_file}_probability.npy")
binary_path = os.path.join(xline_binary_save_dir, f"{example_xline_file}_binary.npy")

prediction_prob = np.load(prob_path)
prediction_binary = np.load(binary_path)

print("Example Xline:", example_xline_file)
print("Probability shape:", prediction_prob.shape)
print("Binary shape:", prediction_binary.shape)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(prediction_prob, cmap="jet", aspect="auto", vmin=0, vmax=1)
plt.colorbar()
plt.title("Prediction Probability")
plt.xlabel("Inline")
plt.ylabel("Time Sample")

plt.subplot(1, 2, 2)
plt.imshow(prediction_binary, cmap="Reds", aspect="auto")
plt.title("Binary Prediction")
plt.xlabel("Inline")
plt.ylabel("Time Sample")

plt.tight_layout()
plt.show()