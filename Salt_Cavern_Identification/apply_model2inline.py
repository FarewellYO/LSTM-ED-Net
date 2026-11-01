'''
@author: Jingwen Xue
@date: 2026-08-08
@description:
    This script applies the trained LSTM-EDNet model to all Inline seismic profiles, generates prediction   probability maps and binary predictions, and saves the results for visualization.
'''

#%%
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%% Path Configuration
inline_norm_dir = r"The_path_to_normalized_inline_data"
prob_save_dir = r"The_path_to_save_probability_results"
binary_save_dir = r"The_path_to_save_binary_results"
os.makedirs(prob_save_dir, exist_ok=True)
os.makedirs(binary_save_dir, exist_ok=True)

#%% Load the LSTM-EDNet Model
model_path = r"The_path_to_trained_model.keras"
model = tf.keras.models.load_model(model_path, compile=False)

print("Model loaded successfully")
print(model.input_shape)
print(model.output_shape)

#%% Load a Single Normalized Inline Profile and Generate Predictions
inline_path = r"The_path_to_normalized_inline_file.npy"
inline_norm = np.load(inline_path)
print("Normalized inline shape:",inline_norm.shape)

inline_input = inline_norm.T
print("Trace-wise input:", inline_input.shape)

X=np.expand_dims(inline_input, axis=-1)
print("Model input shape:",X.shape)

prediction_prob=model.predict(X, batch_size=32, verbose=1)
prediction_prob=np.squeeze(prediction_prob)
print("Prediction probability:", prediction_prob.shape)

threshold=0.5
prediction_binary=(prediction_prob>=threshold).astype(np.uint8)
print("Binary prediction generated")

#%% Visualize Prediction Results for a Single Profile
seismic_img = inline_norm
gt_mask = prediction_binary.T * 0 
prob_map = prediction_prob.T
binary_mask = prediction_binary.T
plt.figure(figsize=(14,10))

# 1 Original seismic
plt.subplot(2,2,1)
plt.imshow(seismic_img, cmap="gray", aspect="auto")
plt.title("Original Seismic")
plt.xlabel("Trace")
plt.ylabel("Time Sample")

# 2 Prediction probability
plt.subplot(2,2,2)
plt.imshow(seismic_img, cmap="gray", aspect="auto")
im_prob = plt.imshow(prob_map, cmap="jet", alpha=0.5, aspect="auto", vmin=0, vmax=1)
plt.title("Prediction Probability")
plt.xlabel("Trace")
plt.ylabel("Time Sample")
plt.colorbar(im_prob,fraction=0.046, pad=0.04)

# 3 Binary prediction
plt.subplot(2,2,3)
plt.imshow(seismic_img,cmap="gray", aspect="auto")
plt.imshow(binary_mask,cmap="Reds",alpha=0.45,aspect="auto")
plt.title("Binary Prediction (Threshold=0.5)")
plt.xlabel("Trace")
plt.ylabel("Time Sample")

# 4 Probability only
plt.subplot(2,2,4)
plt.imshow(prob_map,cmap="jet", aspect="auto", vmin=0,vmax=1)
plt.title("Probability Map")
plt.xlabel("Trace")
plt.ylabel("Time Sample")
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

#%% Retrieve All Inline Profiles
inline_files = sorted([f for f in os.listdir(inline_norm_dir) if f.endswith("_n.npy")])
print(f"A total of {len(inline_files)} Inline profiles found.")

#%% Batch Prediction for All Inline Profiles
threshold = 0.5

for i, file in enumerate(inline_files):

    print(f"Processing {i+1}/{len(inline_files)}: {file}")

    inline_path = os.path.join(inline_norm_dir, file)
    inline_norm = np.load(inline_path)

    print("Inline shape:", inline_norm.shape)

    inline_input = inline_norm.T

    X = np.expand_dims(inline_input, axis=-1)

    prediction_prob = model.predict(X, batch_size=32, verbose=0)

    prediction_prob = np.squeeze(prediction_prob)

    prediction_binary = (prediction_prob >= threshold).astype(np.uint8)

    prediction_prob_save = prediction_prob.T
    prediction_binary_save = prediction_binary.T

    inline_name = file.replace("_n.npy", "")

    np.save(os.path.join(prob_save_dir, f"{inline_name}_probability.npy"), prediction_prob_save)
    np.save(os.path.join(binary_save_dir, f"{inline_name}_binary.npy"), prediction_binary_save)

    print(f"{inline_name} processing completed. Saved shape: {prediction_prob_save.shape}")

print("Batch prediction for all Inline profiles completed!")

#%% View Prediction Results
prob_path = r"The_path_to_probability_file.npy"
binary_path = r"The_path_to_binary_file.npy"

prediction_prob = np.load(prob_path)
prediction_binary = np.load(binary_path)
print("Probability shape:", prediction_prob.shape)
print("Binary shape:", prediction_binary.shape)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(prediction_prob, cmap="jet", aspect="auto", vmin=0, vmax=1)
plt.colorbar()
plt.title("Prediction Probability")
plt.xlabel("Trace")
plt.ylabel("Time Sample")

plt.subplot(1,2,2)
plt.imshow(prediction_binary, cmap="Reds", aspect="auto")
plt.title("Binary Prediction")
plt.xlabel("Trace")
plt.ylabel("Time Sample")
plt.tight_layout()
plt.show()

