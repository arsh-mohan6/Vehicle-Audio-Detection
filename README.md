# Vehicle Audio Detection Using Deep Learning

A deep learning project that classifies vehicle sounds into **Bicycle,
Bus, Car, Motorcycle, and Truck** using audio signal processing and a
dual-branch Convolutional Neural Network (CNN).

The system processes WAV recordings, extracts two complementary audio
representations---**MFCC** and **Mel Spectrogram**---and combines them
to predict the vehicle category.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [Dataset](#dataset)
-   [Project Structure](#project-structure)
-   [Workflow](#workflow)
-   [Audio Preprocessing](#audio-preprocessing)
-   [Data Splitting and Augmentation](#data-splitting-and-augmentation)
-   [Feature Extraction](#feature-extraction)
-   [Model Architecture](#model-architecture)
-   [Training Configuration](#training-configuration)
-   [Results](#results)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Dataset Setup](#dataset-setup)
-   [Running the Project](#running-the-project)
-   [Prediction on a New Audio File](#prediction-on-a-new-audio-file)
-   [Evaluation](#evaluation)
-   [Generated Outputs](#generated-outputs)
-   [Important Notes and Limitations](#important-notes-and-limitations)
-   [Future Improvements](#future-improvements)

------------------------------------------------------------------------

## Project Overview

Vehicle sounds contain distinct acoustic patterns. This project uses
those patterns to identify the type of vehicle from an audio recording.

The pipeline covers the complete machine learning workflow:

1.  Dataset inspection and analysis
2.  Audio preprocessing
3.  Stratified train, validation, and test splitting
4.  Training-only audio augmentation
5.  MFCC and Mel Spectrogram feature extraction
6.  Dual-branch CNN construction
7.  Model training and checkpointing
8.  Prediction on individual WAV files
9.  Final evaluation on the held-out test set

## Key Features

-   Supports five vehicle categories.
-   Processes WAV audio into a consistent format.
-   Uses MFCC and Mel Spectrogram features together.
-   Uses on-the-fly augmentation to balance the training classes.
-   Uses a dual-branch CNN to learn from both feature representations.
-   Saves the best model and training history.
-   Produces class probabilities for individual predictions.
-   Evaluates performance using multiple classification metrics and
    plots.

## Dataset

The project uses **2,568 WAV recordings** across five vehicle
categories.

  Vehicle class     Original recordings
  --------------- ---------------------
  Bicycle                           617
  Bus                               919
  Car                               230
  Motorcycle                        537
  Truck                             265
  **Total**                   **2,568**

### Initial dataset observations

-   All 2,568 audio files were valid during dataset analysis.
-   1,400 files used a sample rate of 44,100 Hz.
-   1,168 files used a sample rate of 48,000 Hz.
-   2,537 recordings were stereo and 31 were mono.
-   Mean duration was approximately 4.997 seconds.
-   Recordings were standardised to five seconds during preprocessing.

> **Dataset availability:** Audio recordings are not included in this
> repository. Place your dataset in the folder structure shown in
> [Dataset Setup](#dataset-setup). Make sure you have permission to use
> and redistribute the recordings.

## Project Structure

The project is organised into separate folders for source code, raw and
processed audio, extracted features, trained models, and results.

``` text
Vehicle-Audio-Detection/
│
├── dataset/
│   ├── raw/
│   │   ├── Bics/                 # Original bicycle recordings
│   │   ├── bus/                  # Original bus recordings
│   │   ├── Cars/                 # Original car recordings
│   │   ├── Motocycles/           # Original motorcycle recordings
│   │   └── Truck/                # Original truck recordings
│   │
│   ├── processed/                # Preprocessed WAV recordings
│   │   ├── Bics/
│   │   ├── bus/
│   │   ├── Cars/
│   │   ├── Motocycles/
│   │   └── Truck/
│   │
│   └── split/
│       ├── train/                # Training recordings
│       ├── validation/           # Validation recordings
│       └── test/                 # Held-out test recordings
│
├── features/
│   ├── mfcc/                     # Extracted MFCC feature arrays
│   ├── mel/                      # Extracted Mel Spectrogram arrays
│   ├── labels/                   # Encoded class labels
│   └── metadata/
│       └── feature_normalization.npz
│
├── models/
│   └── best_vehicle_audio_model.keras
│
├── results/
│   ├── class_distribution.png
│   ├── sample_rate_distribution.png
│   ├── duration_distribution.png
│   ├── audio_metadata.csv
│   ├── preprocessing_report.csv
│   ├── training_log.csv
│   ├── training_history.json
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── overall_metrics.csv
│   ├── class_wise_metrics.csv
│   ├── classification_report.txt
│   ├── evaluation_metrics.json
│   └── test_predictions.csv
│
├── src/
│   ├── A_Dataset_Analysis.py
│   ├── B1_Preprocessing.py
│   ├── B2_Data_Splitting.py
│   ├── B3_Data_Augmentation.py
│   ├── C_Feature_Extraction.py
│   ├── D1_Model_Preparation.py
│   ├── D2_Model_Architecture.py
│   ├── D3_Model_Training.py
│   ├── G1_Training_Visualization.py
│   ├── E_Prediction.py
│   └── F_Evaluation.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Some generated files and folders appear only after the corresponding
stage has been run. Empty directories may be represented in Git using
`.gitkeep` files.*

## Workflow

``` text
Raw WAV Dataset
      |
      v
A. Dataset Analysis
      |
      v
B1. Audio Preprocessing
      |
      v
B2. Train / Validation / Test Split
      |
      +------------------------------+
      |                              |
      v                              v
Training Data                  Validation / Test Data
      |                              |
      v                              |
B3. On-the-Fly Augmentation           |
      |                              |
      v                              |
MFCC + Mel Feature Extraction         |
      |                              |
      v                              |
D1. Training Data Preparation         |
      |                              |
      v                              |
D2. Dual-Branch CNN                   |
      |                              |
      v                              |
D3. Model Training <------------------+
      |
      v
Best Saved Model
      |
      +--------------------+
      |                    |
      v                    v
E. Prediction         F. Test Evaluation
```

## Audio Preprocessing

Every recording is converted into a consistent audio format before
feature extraction.

  Operation               Configuration
  ----------------------- ----------------------------
  Input format            WAV
  Channel conversion      Mono
  Target sample rate      22,050 Hz
  Target duration         5 seconds
  Samples per recording   110,250
  Duration handling       Zero-padding or truncation
  Amplitude handling      Peak normalisation

-   Stereo recordings are converted to mono.
-   Audio is resampled to 22,050 Hz.
-   Short recordings are zero-padded.
-   Long recordings are truncated to five seconds.
-   Peak amplitude normalisation is applied.
-   Processed recordings are saved under `dataset/processed/`.

All 2,568 recordings were successfully processed in the completed run.

## Data Splitting and Augmentation

### Dataset split

A stratified split with `random_state=42` was used to preserve class
proportions.

  Subset         Percentage   Recordings
  ------------ ------------ ------------
  Training              70%        1,797
  Validation            15%          385
  Test                  15%          386
  **Total**        **100%**    **2,568**

### Audio augmentation

The training pipeline uses five augmentation techniques:

1.  **Gaussian noise:** Adds low-level noise to recordings.
2.  **Pitch shifting:** Changes pitch to create acoustic variation.
3.  **Time stretching:** Changes playback speed while aiming to preserve
    pitch.
4.  **Time shifting:** Moves sound events along the time axis.
5.  **Gain adjustment:** Changes recording amplitude.

### Balanced training data

The training generator creates a balanced set of 1,000 training entries
per class.

  -----------------------------------------------------------------------
  Class           Original training         Additional    Total per class
                            entries  augmented entries 
  -------------- ------------------ ------------------ ------------------
  Bicycle                       432                568              1,000

  Bus                           643                357              1,000

  Car                           161                839              1,000

  Motorcycle                    376                624              1,000

  Truck                         185                815              1,000

  **Total**               **1,797**          **3,203**          **5,000**
  -----------------------------------------------------------------------

Augmentation is performed **on the fly** during training rather than
saving every transformed recording as a separate WAV file. The
assignments are shuffled to provide variation across epochs.

**Important:** The 5,000 training entries are not 5,000 independent
original recordings. They consist of the 1,797 original training
recordings and augmented versions derived from them. Validation and test
recordings are not augmented.

## Feature Extraction

Two complementary representations are extracted from each audio
recording.

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

MFCCs represent short-term spectral characteristics of audio using the
Mel frequency scale.

Configuration:

-   Number of coefficients: **40**
-   FFT size (`n_fft`): **2,048**
-   Hop length: **512**
-   Number of frames: **216**

### 2. Mel Spectrogram

The Mel Spectrogram represents the energy distribution of frequencies
over time on the Mel scale.

Configuration:

-   Number of Mel bands: **128**
-   FFT size: **2,048**
-   Hop length: **512**
-   Number of frames: **216**
-   Power values are converted to decibel (dB) values.

### Feature array dimensions

  Split        MFCC                Mel Spectrogram      Labels
  ------------ ------------------- -------------------- -----------
  Train        `(1797, 40, 216)`   `(1797, 128, 216)`   `(1797,)`
  Validation   `(385, 40, 216)`    `(385, 128, 216)`    `(385,)`
  Test         `(386, 40, 216)`    `(386, 128, 216)`    `(386,)`

Normalisation statistics are calculated from the training features and
saved in:

``` text
features/metadata/feature_normalization.npz
```

The same training-derived statistics are reused for validation, test,
and new predictions.

## Model Architecture

The project uses a **Dual-Branch CNN** with two input branches:

-   MFCC branch: input shape `(40, 216, 1)`
-   Mel Spectrogram branch: input shape `(128, 216, 1)`

Each branch learns patterns from its respective audio representation.
The learned feature vectors are then combined by concatenation and
passed through fully connected layers for classification.

### Main architecture components

-   Three convolutional blocks in each branch.
-   Convolution filters: 32, 64, and 128.
-   Batch Normalisation and ReLU activation.
-   Max Pooling and Dropout.
-   Global Average Pooling.
-   Dense layers for feature learning and fusion.
-   Concatenation of MFCC and Mel branch outputs.
-   Final Softmax layer with five output classes.
-   L2 regularisation to help control overfitting.

### Model summary

  Parameter                              Value
  -------------------------- -----------------
  Model type                   Dual-Branch CNN
  Input branches                             2
  Output classes                             5
  Total parameters                     263,429
  Trainable parameters                 261,637
  Non-trainable parameters               1,792

## Training Configuration

  Setting                      Value
  ---------------------------- ---------------------------------
  Framework                    TensorFlow / Keras
  Optimiser                    Adam
  Learning rate (initial)      0.001
  Loss function                Sparse Categorical Crossentropy
  Metric                       Accuracy
  Batch size                   8
  Maximum epochs               30
  Training entries per epoch   5,000
  Validation entries           385
  Early stopping               Patience 8, monitor `val_loss`
  Learning-rate reduction      Patience 3, factor 0.5
  Checkpoint selection         Best validation loss

The best model is saved as:

``` text
models/best_vehicle_audio_model.keras
```

Training logs and history are saved in the `results/` directory.

## Results

### Training and validation

At the final training epoch:

  Metric                   Result
  --------------------- ---------
  Training accuracy        98.12%
  Validation accuracy      98.44%
  Validation loss         0.16113
  Epochs completed             30

### Final test-set evaluation

The final evaluation was performed on **386 held-out test recordings**.

```{=html}
<details>
```
```{=html}
<summary>
```
`<strong>`{=html}Test metrics`</strong>`{=html}
```{=html}
</summary>
```
  Metric                                    Score
  ---------------------------------- ------------
  Test accuracy                        **98.70%**
  Balanced accuracy                        99.17%
  Macro precision                          98.48%
  Macro recall                             99.17%
  Macro F1-score                           98.81%
  Weighted precision                       98.75%
  Weighted recall                          98.70%
  Weighted F1-score                        98.71%
  Macro ROC-AUC                            0.9989
  Macro average precision (AP)             0.9963
  Matthews correlation coefficient         0.9829
  Cohen's kappa                            0.9828
  Log loss                                 0.0876
  Top-2 accuracy                           99.22%

```{=html}
</details>
```
### Test predictions

-   Correct predictions: **381**
-   Incorrect predictions: **5**
-   Total test recordings: **386**

### Class-wise test performance

  Class          Precision    Recall   F1-score   Test support
  ------------ ----------- --------- ---------- --------------
  Bicycle          100.00%   100.00%    100.00%             93
  Bus              100.00%    97.10%     98.53%            138
  Car               97.14%   100.00%     98.55%             34
  Motorcycle        95.24%    98.77%     96.97%             81
  Truck            100.00%   100.00%    100.00%             40

The five errors were:

-   Four Bus recordings predicted as Motorcycle.
-   One Motorcycle recording predicted as Car.

> These are results from this dataset and this particular held-out
> split. They should not be interpreted as a guarantee of the same
> accuracy on arbitrary real-world recordings.

## Requirements

The project uses Python and the following main libraries:

-   NumPy
-   Pandas
-   Librosa
-   SoundFile
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   TensorFlow
-   SciPy
-   Joblib
-   tqdm
-   rich

The completed development environment used TensorFlow 2.21.0. Exact
package versions may vary depending on the Python version and operating
system. The repository's `requirements.txt` is the preferred source for
the environment dependencies.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/arsh-mohan6/Vehicle-Audio-Detection.git
cd Vehicle-Audio-Detection
```

### 2. Create a virtual environment

Windows PowerShell:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

``` cmd
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, check that your Python version is
supported by the TensorFlow version specified in your requirements.

## Dataset Setup

The WAV dataset is not committed to GitHub. Create the following folders
and place the original recordings inside the matching class directory:

``` text
dataset/
└── raw/
    ├── Bics/
    ├── bus/
    ├── Cars/
    ├── Motocycles/
    └── Truck/
```

Use the folder names exactly as expected by the project scripts. The
class-to-label order used by the model is:

``` python
CLASSES = ["Bics", "bus", "Cars", "Motocycles", "Truck"]
```

The corresponding labels are:

    Label Folder         Vehicle
  ------- -------------- ------------
        0 `Bics`         Bicycle
        1 `bus`          Bus
        2 `Cars`         Car
        3 `Motocycles`   Motorcycle
        4 `Truck`        Truck

Make sure the audio files are WAV recordings and that each class folder
contains the correct vehicle category.

## Running the Project

The project includes a menu-driven entry point:

``` bash
python main.py
```

Use the menu to select the stage you want to run. The source scripts are
also available individually in `src/`.

### Recommended first-time pipeline order

Run the stages in this order:

  ----------------------------------------------------------------------------------------
                  Order Stage            Script                           Purpose
  --------------------- ---------------- -------------------------------- ----------------
                      1 A                `A_Dataset_Analysis.py`          Inspect the raw
                                                                          dataset

                      2 B1               `B1_Preprocessing.py`            Standardise
                                                                          audio recordings

                      3 B2               `B2_Data_Splitting.py`           Create train,
                                                                          validation, and
                                                                          test sets

                      4 B3               `B3_Data_Augmentation.py`        Test
                                                                          augmentation
                                                                          operations

                      5 C                `C_Feature_Extraction.py`        Extract MFCC and
                                                                          Mel features

                      6 D                `D3_Model_Training.py`           Prepare, build,
                                                                          and train the
                                                                          model

                      7 G1               `G1_Training_Visualization.py`   Plot training
                                                                          accuracy and
                                                                          loss

                      8 F                `F_Evaluation.py`                Evaluate the
                                                                          saved model on
                                                                          the test set
  ----------------------------------------------------------------------------------------

**Why is B3 not a separate mandatory step before training?** The
training preparation pipeline performs augmentation dynamically while
loading training samples. Run B3 separately when you want to test or
inspect the augmentation functions.

`D3_Model_Training.py` uses the model preparation and architecture
components. Run the stages in order so that the processed audio, split
data, features, and normalisation statistics exist before training.

> **Important:** Running the training stage again will train the model
> again and may replace the saved best-model checkpoint. The evaluation
> stage expects a trained model and the matching test features.

## Prediction on a New Audio File

After training, use the prediction script:

``` bash
python src/E_Prediction.py
```

Follow the script's prompt for the WAV file path.

The prediction pipeline:

1.  Loads the WAV recording.
2.  Converts it to mono and resamples it to 22,050 Hz.
3.  Pads or truncates it to five seconds.
4.  Extracts MFCC and Mel Spectrogram features.
5.  Applies the saved training normalisation statistics.
6.  Loads the trained CNN model.
7.  Predicts the vehicle class and displays class probabilities.

Example output from one completed prediction:

``` text
PREDICTION RESULT

Vehicle: Motocycles
Confidence: 99.97%

All class probabilities:
Bics: 0.00%
bus: 0.00%
Cars: 0.00%
Motocycles: 99.97%
Truck: 0.03%
```

This is an example prediction for one audio recording. It is not the
overall test accuracy.

## Evaluation

Run:

``` bash
python src/F_Evaluation.py
```

The evaluation script uses the held-out test data and the saved model to
calculate classification performance.

It generates:

-   Confusion matrix
-   ROC curve
-   Precision-recall curve
-   Overall evaluation metrics
-   Class-wise evaluation metrics
-   Classification report
-   Test predictions

The test set should be used for final evaluation, not repeatedly used to
tune model settings.

## Generated Outputs

  ----------------------------------------------------------------------------------
  Folder                  Output                             Description
  ----------------------- ---------------------------------- -----------------------
  `results/`              `class_distribution.png`           Number of recordings
                                                             per class

  `results/`              `sample_rate_distribution.png`     Original sample-rate
                                                             distribution

  `results/`              `duration_distribution.png`        Audio duration
                                                             distribution

  `results/`              `audio_metadata.csv`               Raw dataset metadata

  `results/`              `preprocessing_report.csv`         Preprocessing summary

  `features/metadata/`    `feature_normalization.npz`        Training-derived
                                                             normalisation
                                                             statistics

  `models/`               `best_vehicle_audio_model.keras`   Saved trained model

  `results/`              `training_log.csv`                 Epoch-level training
                                                             log

  `results/`              `training_history.json`            Training history

  `results/`              `accuracy_curve.png`               Training and validation
                                                             accuracy

  `results/`              `loss_curve.png`                   Training and validation
                                                             loss

  `results/`              `confusion_matrix.png`             Test-set confusion
                                                             matrix

  `results/`              `roc_curve.png`                    ROC curves

  `results/`              `precision_recall_curve.png`       Precision-recall curves

  `results/`              `overall_metrics.csv`              Overall test metrics

  `results/`              `class_wise_metrics.csv`           Per-class test metrics

  `results/`              `classification_report.txt`        Text classification
                                                             report

  `results/`              `evaluation_metrics.json`          Machine-readable
                                                             metrics

  `results/`              `test_predictions.csv`             Actual and predicted
                                                             test labels
  ----------------------------------------------------------------------------------

## Important Notes and Limitations

-   **Dataset dependency:** The raw WAV dataset is not included in this
    repository. You must supply it locally.
-   **Reproducibility:** A stratified 70/15/15 split and random state 42
    were used in the completed experiment. Exact reproducibility can
    also depend on library versions, random seeds, and hardware.
-   **Augmentation:** Augmented training examples are derived from
    original training recordings; they are not independent recordings.
-   **Generalisation:** The reported test score applies to the current
    dataset and split. More testing on recordings from different roads,
    microphones, distances, and environments is needed before making
    real-world performance claims.
-   **Possible recording overlap:** Duplicate or near-duplicate
    recordings across the random train, validation, and test split were
    not independently ruled out. A source-grouped split and duplicate
    audit would provide a stronger generalisation check.
-   **Small class supports:** The test set contains 34 Car and 40 Truck
    recordings, so class-wise scores for these categories can change
    noticeably with a different test sample.
-   **Hardware:** The completed training run used CPU execution.
    Training time will depend on the machine and TensorFlow build.

## Future Improvements

-   Evaluate on independently collected road and microphone recordings.
-   Audit the dataset for duplicate and near-duplicate audio files.
-   Use source-grouped splitting where recording-source information is
    available.
-   Test performance under different noise levels and recording
    distances.
-   Explore transfer learning with pretrained audio models.
-   Add a live microphone inference interface.
-   Package the prediction pipeline into a simple web or desktop
    application.

------------------------------------------------------------------------

## Author

**Arsh Mohan Nishant**

GitHub: [arsh-mohan6](https://github.com/arsh-mohan6)

Project repository:
[Vehicle-Audio-Detection](https://github.com/arsh-mohan6/Vehicle-Audio-Detection)

------------------------------------------------------------------------

*Built as a deep learning project for audio-based vehicle
classification.*
