
import os
import random
import librosa
import numpy as np

from tensorflow.keras.utils import Sequence

from B3_Data_Augmentation import (
    add_noise,
    pitch_shift,
    time_stretch,
    time_shift,
    adjust_gain,
    fix_length,
    SAMPLE_RATE,
    TARGET_LENGTH
)


# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = "dataset/split"
FEATURE_PATH = "features"

CLASSES = [
    "Bics",
    "bus",
    "Cars",
    "Motocycles",
    "Truck"
]

TARGET_SAMPLES = 1000
BATCH_SIZE = 8

N_FFT = 2048
HOP_LENGTH = 512

N_MFCC = 40
N_MELS = 128

NORMALIZATION_PATH = os.path.join(
    FEATURE_PATH,
    "metadata",
    "feature_normalization.npz"
)


# ==========================================
# 2. AUGMENTATION TECHNIQUES
# ==========================================

AUGMENTATION_TECHNIQUES = {
    "noise": add_noise,
    "pitch": pitch_shift,
    "stretch": time_stretch,
    "shift": time_shift,
    "gain": adjust_gain
}


# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================

def extract_features(audio):

    audio = np.asarray(audio, dtype=np.float32).squeeze()

    if audio.ndim != 1:
        audio = audio.reshape(-1)

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

    if mfcc.shape != (N_MFCC, 216):
        raise ValueError(f"Incorrect MFCC shape: {mfcc.shape}")

    if mel.shape != (N_MELS, 216):
        raise ValueError(f"Incorrect Mel shape: {mel.shape}")

    return (
        mfcc.astype(np.float32),
        mel.astype(np.float32)
    )


# ==========================================
# 4. CALCULATE TRAINING FEATURE STATISTICS
# ==========================================

def calculate_statistics(feature_type):

    feature_file = os.path.join(
        FEATURE_PATH,
        feature_type,
        "train.npy"
    )

    if not os.path.isfile(feature_file):
        raise FileNotFoundError(
            f"Training feature file not found: {feature_file}"
        )

    data = np.load(
        feature_file,
        mmap_mode="r"
    )

    print(f"\nCalculating {feature_type.upper()} statistics...")

    total_sum = np.zeros(
        data.shape[1],
        dtype=np.float64
    )

    total_count = 0

    # First pass: calculate mean
    for start in range(0, len(data), 32):

        batch = np.asarray(
            data[start:start + 32],
            dtype=np.float64
        )

        total_sum += np.sum(
            batch,
            axis=(0, 2)
        )

        total_count += batch.shape[0] * batch.shape[2]

    mean = total_sum / total_count

    # Second pass: calculate variance
    squared_sum = np.zeros(
        data.shape[1],
        dtype=np.float64
    )

    for start in range(0, len(data), 32):

        batch = np.asarray(
            data[start:start + 32],
            dtype=np.float64
        )

        difference = batch - mean[None, :, None]

        squared_sum += np.sum(
            difference ** 2,
            axis=(0, 2)
        )

    variance = squared_sum / total_count
    std = np.sqrt(variance)

    # Avoid division by zero
    std[std < 1e-8] = 1.0

    return (
        mean.astype(np.float32),
        std.astype(np.float32)
    )


# ==========================================
# 5. LOAD OR CREATE NORMALIZATION STATISTICS
# ==========================================

def get_feature_statistics():

    os.makedirs(
        os.path.dirname(NORMALIZATION_PATH),
        exist_ok=True
    )

    if os.path.isfile(NORMALIZATION_PATH):

        print("\nLoading saved normalization statistics...")

        saved = np.load(NORMALIZATION_PATH)

        mfcc_mean = saved["mfcc_mean"]
        mfcc_std = saved["mfcc_std"]

        mel_mean = saved["mel_mean"]
        mel_std = saved["mel_std"]

    else:

        print("\nCreating normalization statistics")
        print("Using original training features only.")

        mfcc_mean, mfcc_std = calculate_statistics("mfcc")
        mel_mean, mel_std = calculate_statistics("mel")

        np.savez(
            NORMALIZATION_PATH,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            mel_mean=mel_mean,
            mel_std=mel_std
        )

        print("\nNormalization statistics saved.")

    if mfcc_mean.shape != (N_MFCC,):
        raise ValueError("Incorrect MFCC normalization shape.")

    if mel_mean.shape != (N_MELS,):
        raise ValueError("Incorrect Mel normalization shape.")

    print("MFCC statistics shape:", mfcc_mean.shape)
    print("Mel statistics shape:", mel_mean.shape)

    return (
        mfcc_mean,
        mfcc_std,
        mel_mean,
        mel_std
    )


# ==========================================
# 6. NORMALIZE FEATURES
# ==========================================

def normalize_features(mfcc, mel, statistics):

    mfcc_mean, mfcc_std, mel_mean, mel_std = statistics

    # Handle individual samples (2D)
    if mfcc.ndim == 2:

        mfcc = (
            mfcc - mfcc_mean[:, None]
        ) / mfcc_std[:, None]

        mel = (
            mel - mel_mean[:, None]
        ) / mel_std[:, None]

    # Handle batches (3D)
    elif mfcc.ndim == 3:

        mfcc = (
            mfcc - mfcc_mean[None, :, None]
        ) / mfcc_std[None, :, None]

        mel = (
            mel - mel_mean[None, :, None]
        ) / mel_std[None, :, None]

    else:
        raise ValueError(
            f"Unexpected feature dimensions: {mfcc.shape}"
        )

    return (
        mfcc.astype(np.float32),
        mel.astype(np.float32)
    )


# ==========================================
# 7. LOAD ORIGINAL TRAINING FILES
# ==========================================

def load_training_files():

    class_files = {}

    for class_index, class_name in enumerate(CLASSES):

        folder = os.path.join(
            DATASET_PATH,
            "train",
            class_name
        )

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Training folder not found: {folder}"
            )

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ])

        if len(files) == 0:
            raise ValueError(
                f"No WAV files found in {folder}"
            )

        class_files[class_name] = [
            os.path.join(folder, filename)
            for filename in files
        ]

    return class_files


# ==========================================
# 8. CREATE BALANCED TRAINING SAMPLES
# ==========================================

def create_balanced_samples(class_files):

    samples = []

    print("\nCreating balanced training dataset:")

    for class_index, class_name in enumerate(CLASSES):

        original_files = class_files[class_name]
        original_count = len(original_files)

        if original_count > TARGET_SAMPLES:
            raise ValueError(
                f"{class_name} has more than "
                f"{TARGET_SAMPLES} original files."
            )

        # Keep every original sample
        for file_path in original_files:

            samples.append(
                (file_path, class_index, None)
            )

        # Calculate additional samples needed
        additional_count = TARGET_SAMPLES - original_count

        # Assign augmentation techniques evenly
        techniques = list(AUGMENTATION_TECHNIQUES.keys())
        random.shuffle(techniques)

        for i in range(additional_count):

            source_file = random.choice(original_files)
            technique = techniques[i % len(techniques)]

            samples.append(
                (source_file, class_index, technique)
            )

        print(
            f"{class_name}: "
            f"{original_count} original + "
            f"{additional_count} augmented = "
            f"{TARGET_SAMPLES}"
        )

    random.shuffle(samples)

    return samples


# ==========================================
# 9. TRAINING DATA GENERATOR
# ==========================================

class TrainingGenerator(Sequence):

    def __init__(
        self,
        class_files,
        statistics=None,
        batch_size=BATCH_SIZE
    ):

        super().__init__()

        self.class_files = class_files
        self.batch_size = batch_size

        if statistics is None:
            statistics = get_feature_statistics()

        self.statistics = statistics
        self.samples = []

        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil(
            len(self.samples) / self.batch_size
        ))

    def __getitem__(self, index):

        start = index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.samples)
        )

        batch_samples = self.samples[start:end]

        batch_mfcc = []
        batch_mel = []
        batch_labels = []

        for file_path, label, technique in batch_samples:

            # Load original audio
            audio, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            audio = fix_length(audio)

            # Apply augmentation only to additional samples
            if technique is not None:

                audio = AUGMENTATION_TECHNIQUES[technique](audio)

            # Ensure one-dimensional audio
            audio = np.asarray(
                audio,
                dtype=np.float32
            ).squeeze()

            if audio.ndim != 1:
                audio = audio.reshape(-1)

            # Ensure exactly 5 seconds
            audio = fix_length(audio)

            if len(audio) != TARGET_LENGTH:
                raise ValueError(
                    f"Incorrect audio length: {len(audio)}"
                )

            # Extract features
            mfcc, mel = extract_features(audio)

            # Normalize individual 2D features
            mfcc, mel = normalize_features(
                mfcc,
                mel,
                self.statistics
            )

            if mfcc.shape != (N_MFCC, 216):
                raise ValueError(
                    f"Incorrect training MFCC: {mfcc.shape}"
                )

            if mel.shape != (N_MELS, 216):
                raise ValueError(
                    f"Incorrect training Mel: {mel.shape}"
                )

            batch_mfcc.append(mfcc)
            batch_mel.append(mel)
            batch_labels.append(label)

        # Convert into model input arrays
        batch_mfcc = np.array(
            batch_mfcc,
            dtype=np.float32
        )[..., np.newaxis]

        batch_mel = np.array(
            batch_mel,
            dtype=np.float32
        )[..., np.newaxis]

        batch_labels = np.array(
            batch_labels,
            dtype=np.int64
        )

        return (
            (batch_mfcc, batch_mel),
            batch_labels
        )

    def on_epoch_end(self):

        # Create fresh augmented samples every epoch
        self.samples = create_balanced_samples(
            self.class_files
        )

        random.shuffle(self.samples)


# ==========================================
# 10. VALIDATION DATA GENERATOR
# ==========================================

class ValidationGenerator(Sequence):

    def __init__(
        self,
        statistics=None,
        batch_size=BATCH_SIZE
    ):

        super().__init__()

        if statistics is None:
            statistics = get_feature_statistics()

        self.statistics = statistics

        # Load saved validation features
        self.mfcc = np.load(
            os.path.join(
                FEATURE_PATH,
                "mfcc",
                "validation.npy"
            ),
            mmap_mode="r"
        )

        self.mel = np.load(
            os.path.join(
                FEATURE_PATH,
                "mel",
                "validation.npy"
            ),
            mmap_mode="r"
        )

        self.labels = np.load(
            os.path.join(
                FEATURE_PATH,
                "labels",
                "validation.npy"
            )
        )

        self.batch_size = batch_size

        if not (
            len(self.mfcc) == len(self.mel) == len(self.labels)
        ):
            raise ValueError(
                "Validation features and labels have different lengths."
            )

    def __len__(self):

        return int(np.ceil(
            len(self.labels) / self.batch_size
        ))

    def __getitem__(self, index):

        start = index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.labels)
        )

        batch_mfcc = np.asarray(
            self.mfcc[start:end],
            dtype=np.float32
        )

        batch_mel = np.asarray(
            self.mel[start:end],
            dtype=np.float32
        )

        # Normalize using training statistics
        batch_mfcc, batch_mel = normalize_features(
            batch_mfcc,
            batch_mel,
            self.statistics
        )

        batch_mfcc = batch_mfcc[..., np.newaxis]
        batch_mel = batch_mel[..., np.newaxis]

        batch_labels = self.labels[start:end].astype(
            np.int64
        )

        return (
            (batch_mfcc, batch_mel),
            batch_labels
        )


# ==========================================
# 11. MAIN FUNCTION - TEST D1
# ==========================================

def main():

    print("\n===================================")
    print(" VEHICLE AUDIO - D1 PREPARATION")
    print("===================================")

    # Load normalization statistics
    statistics = get_feature_statistics()

    # Load original training files
    class_files = load_training_files()

    original_total = sum(
        len(files) for files in class_files.values()
    )

    print("\nOriginal training files:", original_total)

    # Create balanced training generator
    train_generator = TrainingGenerator(
        class_files,
        statistics=statistics
    )

    # Create validation generator
    validation_generator = ValidationGenerator(
        statistics=statistics
    )

    print("\nBalanced training distribution:")

    for class_index, class_name in enumerate(CLASSES):

        count = sum(
            1 for _, label, _ in train_generator.samples
            if label == class_index
        )

        print(f"{class_name}: {count}")

    print("\nTotal training samples:", len(train_generator.samples))
    print("Training batches:", len(train_generator))
    print("Validation samples:", len(validation_generator.labels))
    print("Validation batches:", len(validation_generator))

    # Test one training batch
    (mfcc, mel), labels = train_generator[0]

    print("\nTraining batch test:")

    print("MFCC shape:", mfcc.shape)
    print("Mel shape:", mel.shape)
    print("Labels shape:", labels.shape)

    print("MFCC batch mean:", float(np.mean(mfcc)))
    print("MFCC batch std:", float(np.std(mfcc)))

    print("Mel batch mean:", float(np.mean(mel)))
    print("Mel batch std:", float(np.std(mel)))

    # Test one validation batch
    (val_mfcc, val_mel), val_labels = validation_generator[0]

    print("\nValidation batch test:")

    print("MFCC shape:", val_mfcc.shape)
    print("Mel shape:", val_mel.shape)
    print("Labels shape:", val_labels.shape)

    print("\nNormalization file:", NORMALIZATION_PATH)

    print("\n===================================")
    print(" D1 PREPARATION COMPLETED!")
    print("===================================")


if __name__ == "__main__":
    main()
