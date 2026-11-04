
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from D1_Model_Preparation import (
    load_training_files,
    TrainingGenerator,
    ValidationGenerator,
    get_feature_statistics
)

from D2_Model_Architecture import build_model


# ==========================================
# 1. TRAINING SETTINGS
# ==========================================

BATCH_SIZE = 8
EPOCHS = 30

LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 8
LR_PATIENCE = 3

MODEL_PATH = "models/best_vehicle_audio_model.keras"

LOG_PATH = "results/training_log.csv"

HISTORY_PATH = "results/training_history.json"


# ==========================================
# 2. CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 3. PREPARE TRAINING DATA
# ==========================================

def prepare_data():

    print("\n===================================")
    print(" LOADING TRAINING DATA")
    print("===================================")

    # Load original training files
    class_files = load_training_files()

    # Load training normalization statistics
    statistics = get_feature_statistics()

    # Create training generator
    train_generator = TrainingGenerator(
        class_files,
        statistics=statistics,
        batch_size=BATCH_SIZE
    )

    # Create validation generator
    validation_generator = ValidationGenerator(
        statistics=statistics,
        batch_size=BATCH_SIZE
    )

    print("\nTraining samples:", len(train_generator.samples))
    print("Training batches:", len(train_generator))

    print("\nValidation samples:", len(validation_generator.labels))
    print("Validation batches:", len(validation_generator))

    return train_generator, validation_generator


# ==========================================
# 4. BUILD AND COMPILE MODEL
# ==========================================

def compile_model():

    print("\n===================================")
    print(" BUILDING MODEL")
    print("===================================")

    # Import architecture from D2
    model = build_model()

    # Compile model
    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("\nModel compiled successfully!")

    return model


# ==========================================
# 5. CREATE CALLBACKS
# ==========================================

def create_callbacks():

    # Save model with the lowest validation loss
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # Stop training when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    # Reduce learning rate when validation loss stops improving
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=0.000001,
        mode="min",
        verbose=1
    )

    # Save epoch-wise training results
    csv_logger = CSVLogger(
        LOG_PATH,
        append=False
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ]


# ==========================================
# 6. TRAIN MODEL
# ==========================================

def train_model():

    print("\n===================================")
    print(" VEHICLE AUDIO MODEL TRAINING")
    print("===================================")

    # Prepare datasets
    train_generator, validation_generator = prepare_data()

    # Build and compile model
    model = compile_model()

    # Create callbacks
    callbacks = create_callbacks()

    print("\n===================================")
    print(" TRAINING CONFIGURATION")
    print("===================================")

    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Early stopping patience:", EARLY_STOPPING_PATIENCE)
    print("LR patience:", LR_PATIENCE)

    print("\nStarting model training...\n")

    # Start training
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ======================================
    # 7. SAVE TRAINING HISTORY
    # ======================================

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w") as file:
        json.dump(history_data, file, indent=4)

    # ======================================
    # 8. FIND BEST TRAINING RESULTS
    # ======================================

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    best_val_loss = float(
        min(history.history["val_loss"])
    )

    best_val_accuracy = float(
        history.history["val_accuracy"][best_epoch - 1]
    )

    # ======================================
    # 9. DISPLAY FINAL RESULTS
    # ======================================

    print("\n===================================")
    print(" TRAINING COMPLETED")
    print("===================================")

    print("Epochs completed:", len(history.history["loss"]))

    print("Best epoch:", best_epoch)

    print("Best validation loss:",
          round(best_val_loss, 4))

    print("Validation accuracy at best epoch:",
          round(best_val_accuracy * 100, 2), "%")

    print("\nSaved files:")
    print("Model:", MODEL_PATH)
    print("Training log:", LOG_PATH)
    print("Training history:", HISTORY_PATH)

    print("\n===================================")
    print(" D3 MODEL TRAINING COMPLETED!")
    print("===================================")


# ==========================================
# 10. MAIN
# ==========================================

if __name__ == "__main__":

    train_model()
