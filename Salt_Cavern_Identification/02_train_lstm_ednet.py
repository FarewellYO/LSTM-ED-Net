
"""
@author: Jingwen Xue
@date: 2026-06-03
@description:
Implement and train the LSTM-ED-Net model for seismic salt cavern
identification using synthetic seismic datasets.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, MaxPooling1D, UpSampling1D, TimeDistributed, Concatenate)
from tensorflow.keras.metrics import (BinaryAccuracy, Precision, Recall, AUC)


# Path Configuration
if "__file__" in globals():
    data_dir = Path(__file__).resolve().parent
else:
    data_dir = Path.cwd()


# Training Configuration
EPOCHS = 2
BATCH_SIZE = 32
BASE_UNITS = 64
THRESHOLD = 0.5


# Load Seismic Datasets
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
                f"Data and label shapes do not match: "
                f"{X.shape} vs {Y.shape}"
            )

        # Convert data to (number of traces, 512 samples)
        if X.shape[0] == 512 and X.shape[1] != 512:
            X = X.T
            Y = Y.T
        elif X.shape[1] == 512:
            pass
        else:
            raise ValueError(
                f"Expected 512 time samples, but received {X.shape}."
            )

        X = np.expand_dims(X, axis=-1).astype(np.float32)
        Y = np.expand_dims((Y > 0.5).astype(np.float32), axis=-1)

        return X, Y

    X_train, Y_train = _load_one_pair("X_train.npy", "Y_train.npy")
    X_val, Y_val = _load_one_pair("X_val.npy", "Y_val.npy")
    X_test, Y_test = _load_one_pair("X_test.npy", "Y_test.npy")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


X_train, Y_train, X_val, Y_val, X_test, Y_test = load_seismic_data(data_dir)

print("\nSeismic datasets loaded successfully.")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# Define Evaluation Metrics
class BinaryF1Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    def result(self):
        return 2.0 * self.tp / (
            2.0 * self.tp + self.fp + self.fn +
            tf.keras.backend.epsilon()
        )

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class BinaryIoUScore(tf.keras.metrics.Metric):

    def __init__(self, name="iou", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    def result(self):
        return self.tp / (
            self.tp + self.fp + self.fn +
            tf.keras.backend.epsilon()
        )

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class BinaryDiceScore(tf.keras.metrics.Metric):

    def __init__(self, name="dice", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    def result(self):
        return 2.0 * self.tp / (
            2.0 * self.tp + self.fp + self.fn +
            tf.keras.backend.epsilon()
        )

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


# Build the LSTM-EDNet Model
def build_lstm_ednet(seq_len, n_features=1, base_units=64):

    inputs = Input(shape=(seq_len, n_features), name="input")

    # Encoder Stage
    skip1 = LSTM(base_units, return_sequences=True, name="skip1")(inputs)
    x = MaxPooling1D(2)(skip1)

    skip2 = LSTM(base_units, return_sequences=True, name="skip2")(x)
    x = MaxPooling1D(2)(skip2)

    skip3 = LSTM(base_units, return_sequences=True, name="skip3")(x)
    x = MaxPooling1D(2)(skip3)

    # Bottleneck
    x = LSTM(base_units * 2, return_sequences=True, name="bottleneck")(x)

    # Decoder Stage
    x = UpSampling1D(2)(x)
    x = Concatenate(axis=-1, name="concat3")([x, skip3])
    x = LSTM(base_units, return_sequences=True)(x)

    x = UpSampling1D(2)(x)
    x = Concatenate(axis=-1, name="concat2")([x, skip2])
    x = LSTM(base_units, return_sequences=True)(x)

    x = UpSampling1D(2)(x)
    x = Concatenate(axis=-1, name="concat1")([x, skip1])
    x = LSTM(base_units, return_sequences=True)(x)

    outputs = TimeDistributed(Dense(1, activation="sigmoid"))(x)

    model = Model(inputs, outputs, name="UNet_LSTM_EDNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
            BinaryF1Score(name="f1_score"),
            BinaryIoUScore(name="iou"),
            BinaryDiceScore(name="dice")])

    return model


# Model Initialization and Training
seq_len = X_train.shape[1]
features = X_train.shape[2]

print(
    f"\nDataset ready: Number of time samples={seq_len}, "
    f"Number of features={features}")

salt_model = build_lstm_ednet(
    seq_len,
    features,
    base_units=BASE_UNITS)

salt_model.summary()

print("\nLSTM-EDNet training started...")

history = salt_model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1)

# Save Trained Model
model_save_path = data_dir / "lstm_ednet_model.keras"

salt_model.save(str(model_save_path))

print(f"\nModel saved to: {model_save_path}")


# Dataset Evaluation Metrics
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

    metric_names = ["loss", "accuracy", "precision", "recall",
        "auc", "f1_score", "iou", "dice"]

    print("\n" + "=" * 80)
    print("Detailed Evaluation Metrics")
    print("=" * 80)

    for name in metric_names:
        print(
            f"{name.upper():<12}"
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


train_metrics = calculate_binary_metrics(salt_model, X_train, Y_train, threshold=THRESHOLD)

val_metrics = calculate_binary_metrics(salt_model, X_val, Y_val, threshold=THRESHOLD)

test_metrics = calculate_binary_metrics(salt_model, X_test, Y_test, threshold=THRESHOLD)

print_metrics_table(train_metrics, val_metrics, test_metrics)

