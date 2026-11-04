
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
    top_k_accuracy_score
)

# SETTINGS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["Bics", "bus", "Cars", "Motocycles", "Truck"]
NUM_CLASSES = len(CLASSES)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_vehicle_audio_model.keras")

MFCC_PATH = os.path.join(BASE_DIR, "features", "mfcc", "test.npy")
MEL_PATH = os.path.join(BASE_DIR, "features", "mel", "test.npy")
LABEL_PATH = os.path.join(BASE_DIR, "features", "labels", "test.npy")

NORM_PATH = os.path.join(
    BASE_DIR, "features", "metadata", "feature_normalization.npz"
)

RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# LOAD TEST DATA
def load_test_data():

    print("Loading test data...")

    mfcc = np.load(MFCC_PATH)
    mel = np.load(MEL_PATH)
    labels = np.load(LABEL_PATH)

    stats = np.load(NORM_PATH)

    mfcc_mean = stats["mfcc_mean"]
    mfcc_std = stats["mfcc_std"]
    mel_mean = stats["mel_mean"]
    mel_std = stats["mel_std"]

    # Same normalization used during training
    mfcc = (mfcc - mfcc_mean[None, :, None]) / mfcc_std[None, :, None]
    mel = (mel - mel_mean[None, :, None]) / mel_std[None, :, None]

    # Add channel dimension
    mfcc = mfcc[..., np.newaxis]
    mel = mel[..., np.newaxis]

    return mfcc, mel, labels


# CONFUSION MATRICES
def save_confusion_matrices(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    # Percentage normalization by actual class
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        ax=axes[0],
        cbar=False
    )

    axes[0].set_title("Confusion Matrix - Counts", fontsize=15)
    axes[0].set_xlabel("Predicted Vehicle")
    axes[0].set_ylabel("Actual Vehicle")

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        vmin=0,
        vmax=100,
        ax=axes[1],
        cbar_kws={"label": "Percentage (%)"}
    )

    axes[1].set_title("Normalized Confusion Matrix (%)", fontsize=15)
    axes[1].set_xlabel("Predicted Vehicle")
    axes[1].set_ylabel("Actual Vehicle")

    plt.suptitle("Vehicle Audio Detection - Confusion Matrices", fontsize=18)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: confusion_matrix.png")

    return cm


# ROC CURVE
def save_roc_curve(y_true, probabilities):

    y_onehot = np.eye(NUM_CLASSES)[y_true]

    plt.figure(figsize=(9, 7))

    for i, name in enumerate(CLASSES):

        fpr, tpr, _ = roc_curve(
            y_onehot[:, i],
            probabilities[:, i]
        )

        score = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {score:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "roc_curve.png"),
        dpi=300
    )

    plt.close()

    print("Saved: roc_curve.png")


# PRECISION-RECALL CURVE
def save_pr_curve(y_true, probabilities):

    y_onehot = np.eye(NUM_CLASSES)[y_true]

    plt.figure(figsize=(9, 7))

    for i, name in enumerate(CLASSES):

        precision, recall, _ = precision_recall_curve(
            y_onehot[:, i],
            probabilities[:, i]
        )

        score = average_precision_score(
            y_onehot[:, i],
            probabilities[:, i]
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP = {score:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "precision_recall_curve.png"),
        dpi=300
    )

    plt.close()

    print("Saved: precision_recall_curve.png")


# MAIN EVALUATION
def evaluate_model():

    print("\nLoading model...")

    model = load_model(MODEL_PATH, compile=False)

    mfcc, mel, y_true = load_test_data()

    print("\nTest samples:", len(y_true))

    print("Running predictions...")

    probabilities = model.predict(
        [mfcc, mel],
        batch_size=8,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    loss = log_loss(
        y_true,
        probabilities,
        labels=list(range(NUM_CLASSES))
    )

    top2 = top_k_accuracy_score(
        y_true,
        probabilities,
        k=2,
        labels=list(range(NUM_CLASSES))
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="macro"
    )

    pr_auc = average_precision_score(
        np.eye(NUM_CLASSES)[y_true],
        probabilities,
        average="macro"
    )

    # Specificity from confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    specificity = []

    for i in range(NUM_CLASSES):

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        spec = tn / (tn + fp) if (tn + fp) else 0
        specificity.append(spec)

    # Print overall results
    print("\n====================================")
    print("       OVERALL MODEL EVALUATION")
    print("====================================")

    overall = {
        "Test Samples": len(y_true),
        "Correct Predictions": int(np.sum(y_true == y_pred)),
        "Incorrect Predictions": int(np.sum(y_true != y_pred)),
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1": weighted_f1,
        "Macro ROC-AUC": roc_auc,
        "Macro PR-AUC (Average Precision)": pr_auc,
        "MCC": mcc,
        "Cohen Kappa": kappa,
        "Log Loss": loss,
        "Top-2 Accuracy": top2
    }

    for name, value in overall.items():

        if isinstance(value, float):
            print(f"{name}: {value:.4f} ({value * 100:.2f}%)"
                  if "Accuracy" in name or "Precision" in name
                  or "Recall" in name or "F1" in name
                  else f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    # Class-wise metrics
    class_results = pd.DataFrame({
        "Class": CLASSES,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "Support": support
    })

    print("\nCLASS-WISE RESULTS")
    print(class_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    # Save results
    pd.DataFrame([overall]).to_csv(
        os.path.join(RESULT_DIR, "overall_metrics.csv"),
        index=False
    )

    class_results.to_csv(
        os.path.join(RESULT_DIR, "class_wise_metrics.csv"),
        index=False
    )

    with open(
        os.path.join(RESULT_DIR, "classification_report.txt"),
        "w"
    ) as file:

        file.write("OVERALL METRICS\n\n")

        for name, value in overall.items():
            file.write(f"{name}: {value}\n")

        file.write("\nCLASSIFICATION REPORT\n\n")
        file.write(report)

    with open(
        os.path.join(RESULT_DIR, "evaluation_metrics.json"),
        "w"
    ) as file:

        json.dump(overall, file, indent=4)

    # Save individual predictions
    prediction_data = pd.DataFrame({
        "Actual": [CLASSES[i] for i in y_true],
        "Predicted": [CLASSES[i] for i in y_pred],
        "Confidence": np.max(probabilities, axis=1) * 100,
        "Correct": y_true == y_pred
    })

    prediction_data.to_csv(
        os.path.join(RESULT_DIR, "test_predictions.csv"),
        index=False
    )

    # Generate graphs
    save_confusion_matrices(y_true, y_pred)
    save_roc_curve(y_true, probabilities)
    save_pr_curve(y_true, probabilities)

    print("\n====================================")
    print("       EVALUATION COMPLETED")
    print("====================================")
    print("All reports and graphs saved in results/")


if __name__ == "__main__":
    evaluate_model()
