
import os
import shutil
from sklearn.model_selection import train_test_split

# Dataset paths
DATASET_PATH = "dataset/processed"
OUTPUT_PATH = "dataset/split"

CLASSES = ["Bics", "bus", "Cars", "Motocycles", "Truck"]

# Split settings
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.50
RANDOM_STATE = 42


def main():

    files = []
    labels = []

    # Collect audio files and their labels
    for class_name in CLASSES:

        folder = os.path.join(DATASET_PATH, class_name)

        for file_name in os.listdir(folder):

            if file_name.lower().endswith(".wav"):

                file_path = os.path.join(folder, file_name)

                files.append(file_path)
                labels.append(class_name)

    print("Total audio files:", len(files))

    # First split: 70% training, 30% temporary
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Second split: temporary data into 15% validation and 15% testing
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=temp_labels
    )

    # Save files into separate folders
    splits = {
        "train": (train_files, train_labels),
        "validation": (val_files, val_labels),
        "test": (test_files, test_labels)
    }

    for split_name, (split_files, split_labels) in splits.items():

        for file_path, class_name in zip(split_files, split_labels):

            output_folder = os.path.join(
                OUTPUT_PATH, split_name, class_name
            )

            os.makedirs(output_folder, exist_ok=True)

            shutil.copy2(
                file_path,
                os.path.join(output_folder, os.path.basename(file_path))
            )

    # Print split summary
    print("\n========== DATA SPLITTING SUMMARY ==========")

    for split_name, (split_files, split_labels) in splits.items():

        print(f"\n{split_name.upper()}: {len(split_files)} files")

        for class_name in CLASSES:
            count = split_labels.count(class_name)
            print(f"{class_name}: {count}")

    print("\nData splitting completed successfully!")


if __name__ == "__main__":
    main()
