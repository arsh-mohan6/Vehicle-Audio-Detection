
import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from B3_Data_Augmentation import augment_audio, SAMPLE_RATE

# Test settings
INPUT_FILE = "dataset/split/train/Cars"
OUTPUT_PATH = "results/augmentation_test"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():

    files = [
        f for f in os.listdir(INPUT_FILE)
        if f.lower().endswith(".wav")
    ]

    if not files:
        print("No audio files found!")
        return

    file_path = os.path.join(INPUT_FILE, files[0])

    # Load original audio
    original, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Generate five variations
    variations = []

    for i in range(5):
        augmented = augment_audio(original)
        variations.append(augmented)

        output_file = os.path.join(
            OUTPUT_PATH,
            f"augmented_{i + 1}.wav"
        )

        sf.write(output_file, augmented, sr)

    # Plot original and augmented waveforms
    fig, axes = plt.subplots(6, 1, figsize=(12, 10))

    axes[0].plot(original)
    axes[0].set_title("Original Audio")

    for i, audio in enumerate(variations):

        axes[i + 1].plot(audio)
        axes[i + 1].set_title(f"Augmented Audio {i + 1}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_PATH, "waveform_comparison.png")
    )

    plt.show()

    print("\nAugmentation quality test completed!")
    print("Original file:", file_path)
    print("Generated variations:", len(variations))
    print("Output folder:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
