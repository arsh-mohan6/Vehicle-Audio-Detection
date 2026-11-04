import os
import numpy as np
import librosa
import tensorflow as tf


# ==========================================
# 1. SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_RATE = 22050
DURATION = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_FFT = 2048
HOP_LENGTH = 512

N_MFCC = 40
N_MELS = 128

CLASSES = [
    "Bics",
    "bus",
    "Cars",
    "Motocycles",
    "Truck"
]

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_vehicle_audio_model.keras"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR, "features", "metadata",
    "feature_normalization.npz"
)


# ==========================================
# 2. LOAD NORMALIZATION STATISTICS
# ==========================================

def load_statistics():

    saved = np.load(NORMALIZATION_PATH)

    return (
        saved["mfcc_mean"],
        saved["mfcc_std"],
        saved["mel_mean"],
        saved["mel_std"]
    )


# ==========================================
# 3. AUDIO PREPROCESSING
# ==========================================

def preprocess_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Make audio exactly 5 seconds
    if len(audio) < TARGET_LENGTH:
        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )
    else:
        audio = audio[:TARGET_LENGTH]

    return audio.astype(np.float32)


# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================

def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return (
        mfcc.astype(np.float32),
        mel.astype(np.float32)
    )


# ==========================================
# 5. FEATURE NORMALIZATION
# ==========================================

def normalize_features(mfcc, mel, statistics):

    mfcc_mean, mfcc_std, mel_mean, mel_std = statistics

    mfcc = (
        mfcc - mfcc_mean[:, None]
    ) / mfcc_std[:, None]

    mel = (
        mel - mel_mean[:, None]
    ) / mel_std[:, None]

    return mfcc, mel


# ==========================================
# 6. VEHICLE PREDICTION
# ==========================================

def predict_vehicle(file_path):

    if not os.path.isfile(file_path):
        raise FileNotFoundError("Audio file not found.")

    if not file_path.lower().endswith(".wav"):
        raise ValueError("Please provide a WAV file.")

    print("\nLoading audio...")

    audio = preprocess_audio(file_path)

    print("Extracting MFCC and Mel features...")

    mfcc, mel = extract_features(audio)

    statistics = load_statistics()

    mfcc, mel = normalize_features(
        mfcc,
        mel,
        statistics
    )

    # Add batch and channel dimensions
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    mel = mel[np.newaxis, ..., np.newaxis]

    print("Running prediction...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    probabilities = model.predict(
        [mfcc, mel],
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASSES[predicted_index]

    confidence = float(probabilities[predicted_index]) * 100

    print("\n================================")
    print("       PREDICTION RESULT")
    print("================================")

    print("Vehicle:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll class probabilities:")

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name}: "
            f"{probabilities[i] * 100:.2f}%"
        )

    print("================================")


# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def main():

    print("\nVEHICLE AUDIO DETECTION")

    file_path = input(
        "\nEnter the WAV file path: "
    ).strip().strip('"')

    try:
        predict_vehicle(file_path)

    except Exception as error:
        print("\nPrediction failed:", error)


if __name__ == "__main__":
    main()