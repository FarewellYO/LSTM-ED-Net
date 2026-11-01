'''
@author: Jingwen Xue
@date: 2026-06-03
@description:
This script implements the LSTM-EDNet model for seismic salt cavern identification, including model construction, training, evaluation, and prediction visualization.
'''

#%% 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, MaxPooling1D, UpSampling1D, TimeDistributed, Concatenate
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from matplotlib import pyplot as plt

#%% Load Seismic Datasets
def load_seismic_data(data_dir):
    def _load_one_pair(x_file, y_file):
        X = np.load(os.path.join(data_dir, x_file))
        Y = np.load(os.path.join(data_dir, y_file))

        if X.shape[1] != 512: X = X.T
        if Y.shape[1] != 512: Y = Y.T

        X = np.expand_dims(X, axis=-1).astype(np.float32)
        Y = np.expand_dims(Y, axis=-1).astype(np.float32)
        Y = (Y > 0.5).astype(np.int32)

        return X, Y

    X_train, Y_train = _load_one_pair('train_x_n.npy', 'train_y.npy')
    X_val, Y_val = _load_one_pair('validation_x_n.npy', 'validation_y.npy')
    X_test, Y_test = _load_one_pair('test_x_n.npy', 'test_y.npy')

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

#%% Define Evaluation Metrics
class BinaryF1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score',threshold=0.5, **kwargs):
        super(BinaryF1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    @tf.autograph.experimental.do_not_convert
    def result(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

class BinaryIoUScore(tf.keras.metrics.Metric):
 
    def __init__(self, name='iou', threshold=0.5, **kwargs):
        super(BinaryIoUScore, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    @tf.autograph.experimental.do_not_convert
    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + tf.keras.backend.epsilon())

    @tf.autograph.experimental.do_not_convert
    def reset_state(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)


class BinaryDiceScore(tf.keras.metrics.Metric):

    def __init__(self, name='dice', threshold=0.5, **kwargs):
        super(BinaryDiceScore, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    @tf.autograph.experimental.do_not_convert
    def result(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + tf.keras.backend.epsilon())

    @tf.autograph.experimental.do_not_convert
    def reset_state(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

#%% Build the LSTM-EDNet Model
def build_lstm_ednet(seq_len, n_features=1, base_units=64):

    inputs = Input(shape=(seq_len, n_features), name="input")

    # --- Encoder Stage ---
    # Level 1: 512 -> 256
    skip1 = LSTM(base_units, return_sequences=True, name="skip1")(inputs)
    x = MaxPooling1D(2)(skip1)

    # Level 2: 256 -> 128
    skip2 = LSTM(base_units, return_sequences=True, name="skip2")(x)
    x = MaxPooling1D(2)(skip2)

    # Level 3: 128 -> 64
    skip3 = LSTM(base_units, return_sequences=True, name="skip3")(x)
    x = MaxPooling1D(2)(skip3)

    # --- Bottleneck ---
    x = LSTM(base_units * 2, return_sequences=True, name="bottleneck")(x)

    # --- Decoder Stage ---
    # Level 3 Recovery: 64 -> 128
    x = UpSampling1D(2)(x)
    x = Concatenate(axis=-1, name="concat3")([x, skip3])  
    x = LSTM(base_units, return_sequences=True)(x)

    # Level 2 Recovery: 128 -> 256
    x = UpSampling1D(2)(x)
    x = Concatenate(axis=-1, name="concat2")([x, skip2])  
    x = LSTM(base_units, return_sequences=True)(x)

    # Level 1 Recovery: 256 -> 512
    x = UpSampling1D(2)(x)
    x = Concatenate(axis=-1, name="concat1")([x, skip1])  
    x = LSTM(base_units, return_sequences=True)(x)

    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs, outputs, name="UNet_LSTM_EDNet")

    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            BinaryF1Score(name='f1_score'),
            BinaryIoUScore(name='iou'),
            BinaryDiceScore(name='dice')
        ]
    )

    return model

#%% Model Initialization and Training
data_path = r"The_path_to_dataset_directory"
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_seismic_data(data_path)
seq_len = X_train.shape[1]
features = X_train.shape[2]
print(f"Dataset ready: Number of time samples={seq_len}, Number of features={features}")

salt_model = build_lstm_ednet(
    seq_len,
    features,
    base_units=64,)
salt_model.summary()
print("LSTM-EDNet training started...")

history = salt_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

model_save_path = r"The_path_to_save_model\lstm_ednet_model.keras"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
salt_model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

#%% Dataset Evaluation Metrics Function
def calculate_binary_metrics(model, X, Y, threshold=0.5, batch_size=32):

    Y_prob = model.predict(X, batch_size=batch_size, verbose=0)

    y_true = (Y.reshape(-1) > 0.5).astype(np.int32)
    y_prob = Y_prob.reshape(-1).astype(np.float32)

    y_true_float = y_true.astype(np.float32)
    y_prob_clip = np.clip(y_prob, 1e-8, 1.0 - 1e-8)

    loss = -np.mean(
        y_true_float * np.log(y_prob_clip) +
        (1.0 - y_true_float) * np.log(1.0 - y_prob_clip))

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
        f"FN: {test_metrics['fn']}"
    )
    print("=" * 80)


train_metrics = calculate_binary_metrics(salt_model, X_train, Y_train, threshold=0.5)
val_metrics = calculate_binary_metrics(salt_model, X_val, Y_val, threshold=0.5)
test_metrics = calculate_binary_metrics(salt_model, X_test, Y_test, threshold=0.5)

print_metrics_table(train_metrics, val_metrics, test_metrics)

#%% Prediction Visualization

def visualize_test_results(X_test, Y_test, model, n_traces_per_profile=154):

    print("Generating multi-dimensional analysis results for the test set...")
    Y_pred_prob = model.predict(X_test, batch_size=32)
    Y_pred_binary = (Y_pred_prob > 0.5).astype(np.float32)

    num_profiles = X_test.shape[0] // n_traces_per_profile

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
        plt.title("Ground Truth (Expert)")

        plt.subplot(2, 2, 3)
        plt.imshow(seismic_img, cmap='gray', aspect='auto')
        im_prob = plt.imshow(prob_map, cmap='jet', alpha=0.5, aspect='auto', vmin=0, vmax=1)
        plt.title("Prediction Probability")
        plt.colorbar(im_prob, fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 4)
        plt.imshow(seismic_img, cmap='gray', aspect='auto')
        plt.imshow(binary_mask, cmap='Blues', alpha=0.5, aspect='auto')
        plt.title("Binary Prediction (Thr=0.5)")

        plt.tight_layout()
        plt.show()

visualize_test_results(X_test, Y_test, salt_model)
