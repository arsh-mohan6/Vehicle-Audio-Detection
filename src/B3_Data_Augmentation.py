
import os
import random
import librosa
import soundfile as sf
import numpy as np

# Dataset settings
TRAIN_PATH = "dataset/split/train"

SAMPLE_RATE = 22050
DURATION = 5
TARGET_LENGTH = SAMPLE_RATE * DURATION

CLASSES = ["Bics", "bus", "Cars", "Motocycles", "Truck"]

# Augmentation settings
AUGMENT_PROBABILITY = 0.7


# 1. Add background noise
def add_noise(audio):

    noise = np.random.normal(0, 1, len(audio))

    signal_power = np.mean(audio ** 2)

    if signal_power == 0:
        return audio

    noise_power = np.mean(noise ** 2)

    snr = random.uniform(20, 30)

    scale = np.sqrt(
        signal_power / (noise_power * (10 ** (snr / 10)))
    )

    return audio + noise * scale


# 2. Change pitch
def pitch_shift(audio):

    steps = random.uniform(-1, 1)

    return librosa.effects.pitch_shift(
        y=audio,
        sr=SAMPLE_RATE,
        n_steps=steps
    )


# 3. Change playback speed
def time_stretch(audio):

    rate = random.uniform(0.9, 1.1)

    return librosa.effects.time_stretch(
        y=audio,
        rate=rate
    )


# 4. Shift audio in time
def time_shift(audio):

    shift = random.randint(
        -int(0.2 * SAMPLE_RATE),
        int(0.2 * SAMPLE_RATE)
    )

    shifted = np.zeros_like(audio)

    if shift > 0:
        shifted[shift:] = audio[:-shift]

    elif shift < 0:
        shifted[:shift] = audio[-shift:]

    else:
        shifted = audio.copy()

    return shifted


# 5. Adjust audio volume
def adjust_gain(audio):

    gain_db = random.uniform(-3, 3)

    gain = 10 ** (gain_db / 20)

    return audio * gain


# Keep audio length fixed
def fix_length(audio):

    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio)),
            mode="constant"
        )

    else:
        audio = audio[:TARGET_LENGTH]

    return audio


# Main augmentation function
def augment_audio(audio):

    audio = audio.copy()

    # Sometimes return original audio
    if random.random() > AUGMENT_PROBABILITY:
        return audio

    techniques = [
        add_noise,
        pitch_shift,
        time_stretch,
        time_shift,
        adjust_gain
    ]

    # Randomly select one or two techniques
    count = random.randint(1, 2)

    selected = random.sample(techniques, count)

    for technique in selected:

        audio = technique(audio)

        audio = fix_length(audio)

    return audio.astype(np.float32)


# Test augmentation on one training file
def main():

    test_class = "Cars"

    folder = os.path.join(TRAIN_PATH, test_class)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    if not files:
        print("No audio files found!")
        return

    file_path = os.path.join(folder, files[0])

    # Load original audio
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Apply augmentation
    augmented = augment_audio(audio)

    # Compare original and augmented audio
    print("\n========== AUGMENTATION TEST ==========")

    print("Original length:", len(audio))
    print("Augmented length:", len(augmented))

    print("Original duration:", len(audio) / sr)
    print("Augmented duration:", len(augmented) / sr)

    print("Original peak:", np.max(np.abs(audio)))
    print("Augmented peak:", np.max(np.abs(augmented)))

    print("Original audio shape:", audio.shape)
    print("Augmented audio shape:", augmented.shape)

    print("\nAugmentation test completed!")


if __name__ == "__main__":
    main()
