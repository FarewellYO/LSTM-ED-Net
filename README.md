# LSTM-ED-Net

**A method for seismic identification of salt caverns based on multiscale temporal features**

## 1. Overview

This repository provides the Python implementation of LSTM-ED-Net, a deep learning framework developed for seismic identification of salt caverns based on multiscale temporal features.

The repository contains the following components:

1. Viscoelastic seismic forward modeling for comparing seismic responses under rock salt and brine replacement scenarios.
2. Synthetic dataset preparation and normalization.
3. LSTM-ED-Net model construction and training.
4. Model performance evaluation.
5. Inline and Xline prediction demonstrations.

## 2. Repository Structure

```text
LSTM-ED-Net/
│
├── Salt_Cavern_Identification/
│   ├── 01_prepare_datasets.py
│   ├── 02_train_lstm_ednet.py
│   ├── 03_evaluate_model.py
│   ├── 04_apply_model2inline.py
│   ├── 05_apply_model2xline.py
│   ├── SeismicData.npy
│   └── LabelData.npy
│
├── Viscoelastic_Seismic_Forward_Modeling/
│   ├── 01_build_salt_brine_models.py
│   └── 02_generate_synthetic_seismic_records.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 3. Environment and Dependencies

The codes were developed and tested using Python 3.10.

The required Python packages are listed in `requirements.txt`.

Install all dependencies using:

```bash
python -m pip install -r requirements.txt
```

## 4. Synthetic Dataset

### 4.1 Input Data

The synthetic seismic dataset is provided in the `Salt_Cavern_Identification` directory.

Two input files are required:

- `SeismicData.npy`: Synthetic seismic amplitude data.
- `LabelData.npy`: Corresponding binary labels.

Both files are two-dimensional NumPy arrays with identical dimensions.

The expected data organization is:

- Rows: 512 time samples.
- Columns: Seismic traces.

The seismic data and corresponding labels must have matching dimensions.

### 4.2 Dataset Description and Preparation

The synthetic dataset consists of a two-dimensional seismic profile and its corresponding binary label matrix. The seismic data are stored in `SeismicData.npy`, while the corresponding labels are stored in `LabelData.npy`. Both arrays have identical dimensions, with 512 time samples along each seismic trace. Each column represents an individual seismic trace, and each row corresponds to a time sample. The binary labels identify the target seismic responses associated with the salt cavern and distinguish them from the background seismic signals.

Before model training, the seismic traces are randomly shuffled and divided into training, validation, and test datasets, accounting for 70%, 15%, and 15% of the total traces, respectively. A fixed random seed of 42 is used to ensure reproducibility. Data normalization is performed using the global mean and standard deviation calculated from the training dataset, and the same normalization parameters are applied to the validation and test datasets. The resulting datasets are subsequently used for model training, validation, and performance evaluation.

## 5. Salt Cavern Identification Workflow

The following five scripts constitute the main LSTM-ED-Net identification workflow. They should be executed sequentially.

### Step 1. Dataset Preparation

**Script:** `01_prepare_datasets.py`

This script:

- Loads the synthetic seismic data and corresponding labels.
- Randomly divides the traces into training, validation, and test datasets.
- Calculates global normalization parameters using the training dataset.
- Normalizes all three datasets.
- Saves the processed datasets and normalization parameters.

Generated files include:

- `X_train.npy`
- `Y_train.npy`
- `X_val.npy`
- `Y_val.npy`
- `X_test.npy`
- `Y_test.npy`
- `train_mean.npy`
- `train_std.npy`

### Step 2. LSTM-ED-Net Model Training

**Script:** `02_train_lstm_ednet.py`

This script constructs and trains the LSTM-ED-Net architecture.

The model takes seismic traces containing 512 time samples as input and generates a prediction probability for each time sample.

The trained model is saved as:

`lstm_ednet_model.keras`

### Step 3. Model Performance Evaluation

**Script:** `03_evaluate_model.py`

This script loads the trained model and evaluates its performance on the training, validation, and test datasets.

### Step 4. Inline Seismic Prediction

**Script:** `04_apply_model2inline.py`

This script is designed to apply the trained LSTM-ED-Net model to actual Inline seismic profiles for salt cavern identification. It processes seismic traces along the Inline direction and generates sample-wise prediction probabilities and binary prediction results.

To verify the functionality of the script, the included synthetic test dataset (`X_test.npy` and `Y_test.npy`), generated in Step 1, is used for code execution and testing. Each seismic trace contains 512 time samples, consistent with the input requirements of the network.

### Step 5. Xline Seismic Prediction and Cross-Direction Evaluation

**Script:** `05_apply_model2xline.py`

This script is designed to apply the trained LSTM-EDNet model to actual Xline seismic profiles for cross-direction prediction and evaluation of the model's spatial applicability. It investigates whether a model trained using Inline seismic data can be directly applied to seismic profiles in the orthogonal Xline direction without retraining.

To verify the functionality of the script, the included synthetic test dataset (`X_test.npy` and `Y_test.npy`), generated in Step 1, is used for code execution and testing. The trained model takes the normalized synthetic seismic traces as input and generates corresponding prediction probabilities and binary predictions. The ground-truth labels are used to calculate evaluation metrics.

The synthetic dataset is used exclusively for functional testing. Actual Xline seismic profiles can be used as input to evaluate the cross-direction generalization and spatial applicability of the trained model.

## 6. Viscoelastic Seismic Forward Modeling

The `Viscoelastic_Seismic_Forward_Modeling` directory contains two scripts for constructing geological models and generating synthetic seismic records.

These scripts can be executed independently of the LSTM-ED-Net training workflow.

### Step 1. Construction of Geological Models

**Script:** `01_build_salt_brine_models.py`

This script constructs two geological models:

1. Rock salt model.
2. Brine replacement model.

Lithological properties, including P-wave velocity, density, and quality factor (Q), are assigned to each geological layer.

### Step 2. Synthetic Seismic Record Generation

**Script:** `02_generate_synthetic_seismic_records.py`

This script loads the geological models generated in Step 1 and calculates:

- Interface depths and two-way travel times.
- Elastic reflection coefficients.
- Frequency-dependent complex reflection coefficients.
- Elastic synthetic seismic records.
- Viscoelastic synthetic seismic records considering dispersion and attenuation.

## 7. Running Instructions

**Main Salt Cavern Identification Workflow**

Run the following commands sequentially from the repository root directory:

```bash
python Salt_Cavern_Identification/01_prepare_datasets.py
python Salt_Cavern_Identification/02_train_lstm_ednet.py
python Salt_Cavern_Identification/03_evaluate_model.py
python Salt_Cavern_Identification/04_apply_model2inline.py
python Salt_Cavern_Identification/05_apply_model2xline.py
```

**Viscoelastic Forward Modeling Workflow**

Run the following commands sequentially:

```bash
python Viscoelastic_Seismic_Forward_Modeling/01_build_salt_brine_models.py
python Viscoelastic_Seismic_Forward_Modeling/02_generate_synthetic_seismic_records.py
```

The second script requires the numerical geological model files generated by the first script.

## 8. Notes

1. The included seismic data and labels are synthetic and are provided for demonstrating the code execution and model workflow.
2. All seismic traces used by LSTM-ED-Net contain 512 time samples.
3. The training, validation, and test datasets are generated automatically by the dataset preparation script.
4. The Inline and Xline scripts demonstrate prediction and visualization using synthetic data.

## 9. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



