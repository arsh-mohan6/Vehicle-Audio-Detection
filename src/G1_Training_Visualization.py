
import json
import os
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "results", "training_history.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load training history
with open(HISTORY_PATH, "r") as file:
    history = json.load(file)

# Epoch numbers
epochs = range(1, len(history["accuracy"]) + 1)

# -------------------------------
# 1. Training and Validation Accuracy
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(epochs, history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=300)
plt.show()

# -------------------------------
# 2. Training and Validation Loss
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(epochs, history["loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
plt.show()

print("Training visualization completed!")
print("Saved: results/accuracy_curve.png")
print("Saved: results/loss_curve.png")
