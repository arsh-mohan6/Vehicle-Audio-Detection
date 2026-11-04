
import os
import sys
import subprocess

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

# All project scripts
SCRIPTS = {
    "A": "A_Dataset_Analysis.py",
    "B1": "B1_Preprocessing.py",
    "B2": "B2_Data_Splitting.py",
    "B3": "B3_Data_Augmentation.py",
    "C": "C_Feature_Extraction.py",
    "D": "D3_Model_Training.py",
    "G1": "G1_Training_Visualization.py",
    "E": "E_Prediction.py",
    "F": "F_Evaluation.py"
}


def run_stage(stage):

    script = os.path.join(SRC_DIR, SCRIPTS[stage])

    if not os.path.isfile(script):
        print("\nError: Script not found:", script)
        return

    print("\n" + "=" * 45)
    print("Running Stage:", stage)
    print("Script:", SCRIPTS[stage])
    print("=" * 45)

    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        print("\nStage", stage, "completed successfully!")
    else:
        print("\nStage", stage, "failed!")

    return result.returncode


def run_full_pipeline():

    stages = ["A", "B1", "B2", "C", "D", "G1", "F"]

    print("\nStarting complete ML pipeline...")

    for stage in stages:

        result = run_stage(stage)

        if result != 0:
            print("\nPipeline stopped because a stage failed.")
            return

    print("\nComplete pipeline finished successfully!")


def show_menu():

    print("\n" + "=" * 45)
    print("       VEHICLE AUDIO DETECTION")
    print("=" * 45)

    print("\n1. Dataset Analysis")
    print("2. Audio Preprocessing")
    print("3. Data Splitting")
    print("4. Data Augmentation Test")
    print("5. Feature Extraction")
    print("6. Model Training")
    print("7. Training Visualization")
    print("8. Predict Vehicle")
    print("9. Model Evaluation")
    print("10. Run Complete Pipeline")
    print("0. Exit")

    print("=" * 45)


def main():

    options = {
        "1": "A",
        "2": "B1",
        "3": "B2",
        "4": "B3",
        "5": "C",
        "6": "D",
        "7": "G1",
        "8": "E",
        "9": "F"
    }

    while True:

        show_menu()

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("\nExiting Vehicle Audio Detection.")
            break

        elif choice == "10":
            confirm = input(
                "\nThis will rerun preprocessing and training. Continue? (yes/no): "
            ).strip().lower()

            if confirm == "yes":
                run_full_pipeline()

        elif choice in options:
            run_stage(options[choice])

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
