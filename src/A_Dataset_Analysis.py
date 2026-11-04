import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

# Dataset path
DATASET_PATH = "dataset/raw"

# Vehicle categories
CLASSES = ["Bics", "bus", "Cars", "Motocycles", "Truck"]

# Scan dataset and collect audio file paths
audio_files = []

for category in CLASSES:
    folder_path = os.path.join(DATASET_PATH, category)

    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(".wav")
    ]

    audio_files.extend(files)

    print(f"{category}: {len(files)} files")

print("\nTotal audio files:", len(audio_files))


# Store audio information
audio_data = []

for file_path in audio_files:
    try:
        info = sf.info(file_path)

        audio_data.append({
            "File": os.path.basename(file_path),
            "Class": os.path.basename(os.path.dirname(file_path)),
            "Sample_Rate": info.samplerate,
            "Duration": info.duration,
            "Channels": info.channels
        })

    except Exception as e:
        print("Error reading:", file_path)
        print(e)

# Convert information into a DataFrame
df = pd.DataFrame(audio_data)

print("\nAudio Metadata:")
print(df.head())

print("\nTotal analysed files:", len(df))


# Statistical summary of audio metadata

print("\nDuration Statistics:")
print(df["Duration"].describe())

print("\nSample Rate Distribution:")
print(df["Sample_Rate"].value_counts())

print("\nChannel Distribution:")
print(df["Channels"].value_counts())

print("\nClass-wise Average Duration:")
print(df.groupby("Class")["Duration"].mean())


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Vehicle class distribution
plt.figure(figsize=(9, 5))

sns.countplot(data=df, x="Class")

plt.title("Vehicle Audio Class Distribution")
plt.xlabel("Vehicle Class")
plt.ylabel("Number of Audio Files")
plt.tight_layout()
plt.savefig("results/class_distribution.png")
# plt.show()


# 2. Sample rate distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Sample_Rate")

plt.title("Audio Sample Rate Distribution")
plt.xlabel("Sample Rate (Hz)")
plt.ylabel("Number of Audio Files")
plt.tight_layout()
plt.savefig("results/sample_rate_distribution.png")
# plt.show()


# 3. Audio duration distribution
plt.figure(figsize=(9, 5))

sns.histplot(data=df, x="Duration", bins=30, kde=True)

plt.title("Audio Duration Distribution")
plt.xlabel("Duration (Seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/duration_distribution.png")
# plt.show()


# Check for corrupted audio files

corrupted_files = []

for file_path in audio_files:
    try:
        audio, sr = sf.read(file_path)

        if len(audio) == 0:
            corrupted_files.append(file_path)

    except Exception:
        corrupted_files.append(file_path)

print("\nCorrupted Files:", len(corrupted_files))
print("Valid Files:", len(audio_files) - len(corrupted_files))


# Save audio metadata
df.to_csv("results/audio_metadata.csv", index=False)

print("\nMetadata saved successfully!")