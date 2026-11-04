
import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

# Dataset paths
RAW_PATH = "dataset/raw"
PROCESSED_PATH = "dataset/processed"
RESULTS_PATH = "results"

# Preprocessing settings
SAMPLE_RATE = 22050
DURATION = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION

CLASSES = ["Bics", "bus", "Cars", "Motocycles", "Truck"]


# Create output folders
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# Function to preprocess one audio file
def preprocess_audio(file_path):

    # Load audio and convert to mono
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    # Resample audio
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    # Standardise duration
    if len(audio) < TARGET_LENGTH:
        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio)),
            mode="constant"
        )
    else:
        audio = audio[:TARGET_LENGTH]

    # Peak normalisation
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio


# Main preprocessing process
def main():

    report = []
    total_files = 0
    successful_files = 0
    failed_files = []

    for class_name in CLASSES:

        input_folder = os.path.join(RAW_PATH, class_name)
        output_folder = os.path.join(PROCESSED_PATH, class_name)

        os.makedirs(output_folder, exist_ok=True)

        files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(".wav")
        ]

        print(f"\nProcessing {class_name}: {len(files)} files")

        for file_name in tqdm(files):

            total_files += 1

            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            try:
                audio = preprocess_audio(input_file)

                # Validate processed audio
                if len(audio) != TARGET_LENGTH:
                    raise ValueError("Incorrect audio length")

                if not np.all(np.isfinite(audio)):
                    raise ValueError("Invalid audio values")

                # Save processed audio
                sf.write(output_file, audio, SAMPLE_RATE)

                # Verify saved file
                info = sf.info(output_file)

                if info.samplerate != SAMPLE_RATE:
                    raise ValueError("Incorrect sample rate")

                if info.frames != TARGET_LENGTH:
                    raise ValueError("Incorrect saved duration")

                successful_files += 1

                report.append({
                    "File": file_name,
                    "Class": class_name,
                    "Sample_Rate": info.samplerate,
                    "Duration": info.duration,
                    "Channels": info.channels,
                    "Status": "Success"
                })

            except Exception as e:

                failed_files.append({
                    "File": file_name,
                    "Class": class_name,
                    "Error": str(e)
                })

                report.append({
                    "File": file_name,
                    "Class": class_name,
                    "Status": "Failed",
                    "Error": str(e)
                })

    # Save preprocessing report
    df = pd.DataFrame(report)

    df.to_csv(
        os.path.join(RESULTS_PATH, "preprocessing_report.csv"),
        index=False
    )

    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Total files       : {total_files}")
    print(f"Successful files  : {successful_files}")
    print(f"Failed files      : {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(file)

    print("\nPreprocessing report saved successfully!")


if __name__ == "__main__":
    main()
