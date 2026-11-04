
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. DATASET SETTINGS
# ==========================================

DATASET_PATH = "dataset/split"
OUTPUT_PATH = "features"

CLASSES = [
    "Bics",
    "bus",
    "Cars",
    "Motocycles",
    "Truck"
]

SPLITS = ["train", "validation", "test"]

SAMPLE_RATE = 22050
DURATION = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION

# ==========================================
# 2. FEATURE EXTRACTION SETTINGS
# ==========================================

N_FFT = 2048
HOP_LENGTH = 512

N_MFCC = 40
N_MELS = 128

# ==========================================
# 3. CREATE OUTPUT FOLDERS
# ==========================================

def create_folders():

    folders = [
        "mfcc",
        "mel",
        "labels",
        "metadata"
    ]

    for folder in folders:

        path = os.path.join(OUTPUT_PATH, folder)
        os.makedirs(path, exist_ok=True)


# ==========================================
# 4. LOAD AUDIO
# ==========================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Make every audio file exactly 5 seconds
    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )

    else:

        audio = audio[:TARGET_LENGTH]

    return audio


# ==========================================
# 5. EXTRACT MFCC
# ==========================================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return mfcc.astype(np.float32)


# ==========================================
# 6. EXTRACT MEL SPECTROGRAM
# ==========================================

def extract_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db.astype(np.float32)


# ==========================================
# 7. PROCESS ONE DATASET SPLIT
# ==========================================

def process_split(split):

    print(f"\nProcessing {split.upper()} data...")

    mfcc_data = []
    mel_data = []
    labels = []
    filenames = []
    errors = []

    for class_index, class_name in enumerate(CLASSES):

        folder = os.path.join(
            DATASET_PATH,
            split,
            class_name
        )

        if not os.path.isdir(folder):

            print("Folder not found:", folder)
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ])

        for filename in tqdm(files, desc=class_name):

            file_path = os.path.join(folder, filename)

            try:

                # Load audio
                audio = load_audio(file_path)

                # Extract both features
                mfcc = extract_mfcc(audio)
                mel = extract_mel(audio)

                # Store features
                mfcc_data.append(mfcc)
                mel_data.append(mel)

                labels.append(class_index)

                filenames.append(
                    os.path.join(class_name, filename)
                )

            except Exception as e:

                print(f"\nError processing {file_path}: {e}")

                errors.append({
                    "filename": file_path,
                    "error": str(e)
                })

    # Convert data into NumPy arrays
    mfcc_data = np.array(mfcc_data, dtype=np.float32)
    mel_data = np.array(mel_data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # ======================================
    # 8. SAVE FEATURES
    # ======================================

    np.save(
        os.path.join(OUTPUT_PATH, "mfcc", f"{split}.npy"),
        mfcc_data
    )

    np.save(
        os.path.join(OUTPUT_PATH, "mel", f"{split}.npy"),
        mel_data
    )

    np.save(
        os.path.join(OUTPUT_PATH, "labels", f"{split}.npy"),
        labels
    )

    # Save filename and label mapping
    metadata = pd.DataFrame({
        "filename": filenames,
        "class": [CLASSES[label] for label in labels],
        "label": labels
    })

    metadata.to_csv(
        os.path.join(
            OUTPUT_PATH,
            "metadata",
            f"{split}.csv"
        ),
        index=False
    )

    # Save errors, if any
    if errors:

        error_report = pd.DataFrame(errors)

        error_report.to_csv(
            os.path.join(
                OUTPUT_PATH,
                "metadata",
                f"{split}_errors.csv"
            ),
            index=False
        )

    # ======================================
    # 9. DISPLAY SUMMARY
    # ======================================

    print(f"\n{split.upper()} SUMMARY")

    print("Total successful files:", len(labels))
    print("Failed files:", len(errors))

    print("MFCC shape:", mfcc_data.shape)
    print("Mel shape:", mel_data.shape)
    print("Labels shape:", labels.shape)

    for class_index, class_name in enumerate(CLASSES):

        count = np.sum(labels == class_index)

        print(f"{class_name}: {count}")


# ==========================================
# 10. MAIN FUNCTION
# ==========================================

def main():

    print("\n======================================")
    print("   VEHICLE AUDIO FEATURE EXTRACTION")
    print("======================================")

    create_folders()

    for split in SPLITS:

        process_split(split)

    print("\n======================================")
    print("FEATURE EXTRACTION COMPLETED!")
    print("======================================")

    print("Features saved inside:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
