import warnings
warnings.filterwarnings("ignore")

import random
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score
)
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\DATA")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\MODEL")
RESULTS_DIR = Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\RESULTS")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 40

STAGE1_LR = 1e-4
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-4

FOCAL_GAMMA = 2.0

PATIENCE = 8

NUM_WORKERS = 0
PIN_MEMORY = False

NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 70)
print("OSTEOSENSE V3")
print("EfficientNet-B0 + CLAHE + Weighted Sampling + Focal Loss")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Stage 1 epochs: {STAGE1_EPOCHS}")
print(f"Stage 2 maximum epochs: {STAGE2_EPOCHS}")
print(f"Focal gamma: {FOCAL_GAMMA}")
print("=" * 70)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. CLAHE PREPROCESSING
# ============================================================

def apply_clahe(img):
    """
    Apply CLAHE to the luminance channel of the X-ray.
    """

    img_np = np.array(img)

    if len(img_np.shape) == 2:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        img_np = clahe.apply(img_np)

        return Image.fromarray(img_np).convert("RGB")

    if img_np.shape[2] == 3:

        lab = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGB2LAB
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        lab[:, :, 0] = clahe.apply(
            lab[:, :, 0]
        )

        img_np = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2RGB
        )

        return Image.fromarray(img_np)

    return img.convert("RGB")


# ============================================================
# 4. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Lambda(apply_clahe),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Lambda(apply_clahe),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nDataset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")


# ============================================================
# 6. CHECK CLASS COUNTS
# ============================================================

train_targets = np.array(
    [sample[1] for sample in train_dataset.samples]
)

class_counts = np.bincount(
    train_targets,
    minlength=NUM_CLASSES
)

print("\nTraining class distribution:")

for i, count in enumerate(class_counts):
    print(f"KL {i}: {count}")


# ============================================================
# 7. WEIGHTED RANDOM SAMPLER
# ============================================================

class_weights = 1.0 / class_counts

sample_weights = np.array(
    [class_weights[label] for label in train_targets],
    dtype=np.float64
)

sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print("\nWeightedRandomSampler enabled.")


# ============================================================
# 8. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# ============================================================
# 9. FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=2.0
    ):
        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        inputs,
        targets
    ):

        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


criterion = FocalLoss(
    gamma=FOCAL_GAMMA
)

print("\nLoss:")
print("Focal Loss")
print(f"Gamma = {FOCAL_GAMMA}")


# ============================================================
# 10. LOAD EFFICIENTNET-B0
# ============================================================

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print("\nEfficientNet-B0 loaded.")
print(f"Classifier input features: {num_features}")


# ============================================================
# 11. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    true_labels,
    predictions
):

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    all_labels = []
    all_predictions = []

    running_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    val_loss = (
        running_loss /
        len(val_dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return val_loss, metrics


# ============================================================
# 13. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    epoch_accuracy = (
        correct /
        total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# 14. STAGE 1 — FREEZE BACKBONE
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 — FROZEN BACKBONE")
print("=" * 70)

for parameter in model.features.parameters():
    parameter.requires_grad = False

for parameter in model.classifier.parameters():
    parameter.requires_grad = True


optimizer_stage1 = optim.AdamW(
    model.classifier.parameters(),
    lr=STAGE1_LR,
    weight_decay=WEIGHT_DECAY
)

scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_stage1,
    T_max=STAGE1_EPOCHS,
    eta_min=1e-6
)


best_stage1_f1 = -1.0

stage1_history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_macro_f1": []
}


for epoch in range(STAGE1_EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        optimizer_stage1
    )

    val_loss, val_metrics = validate()

    scheduler_stage1.step()

    stage1_history["train_loss"].append(
        train_loss
    )

    stage1_history["train_accuracy"].append(
        train_accuracy
    )

    stage1_history["val_loss"].append(
        val_loss
    )

    stage1_history["val_accuracy"].append(
        val_metrics["accuracy"]
    )

    stage1_history["val_macro_f1"].append(
        val_metrics["macro_f1"]
    )

    print(
        f"Stage 1 | "
        f"Epoch {epoch + 1}/{STAGE1_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_accuracy * 100:.2f}% | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_metrics['accuracy'] * 100:.2f}% | "
        f"Val Macro F1 {val_metrics['macro_f1']:.4f}"
    )

    if val_metrics["macro_f1"] > best_stage1_f1:

        best_stage1_f1 = (
            val_metrics["macro_f1"]
        )

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_stage1.pth"
        )

        print("  → Best Stage 1 model saved.")


model.load_state_dict(
    torch.load(
        MODEL_DIR / "best_stage1.pth",
        map_location=DEVICE
    )
)

print(
    f"\nBest Stage 1 Macro F1: "
    f"{best_stage1_f1:.4f}"
)


# ============================================================
# 15. STAGE 2 — FULL FINE-TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 — FULL FINE-TUNING")
print("=" * 70)


for parameter in model.features.parameters():
    parameter.requires_grad = True

for parameter in model.classifier.parameters():
    parameter.requires_grad = True


optimizer_stage2 = optim.AdamW(
    [
        {
            "params": model.features.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)


scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_stage2,
    T_max=STAGE2_EPOCHS,
    eta_min=1e-7
)


best_macro_f1 = best_stage1_f1
best_epoch = 0

patience_counter = 0

stage2_history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_macro_f1": []
}


for epoch in range(STAGE2_EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        optimizer_stage2
    )

    val_loss, val_metrics = validate()

    scheduler_stage2.step()

    stage2_history["train_loss"].append(
        train_loss
    )

    stage2_history["train_accuracy"].append(
        train_accuracy
    )

    stage2_history["val_loss"].append(
        val_loss
    )

    stage2_history["val_accuracy"].append(
        val_metrics["accuracy"]
    )

    stage2_history["val_macro_f1"].append(
        val_metrics["macro_f1"]
    )

    print(
        f"Stage 2 | "
        f"Epoch {epoch + 1}/{STAGE2_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_accuracy * 100:.2f}% | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_metrics['accuracy'] * 100:.2f}% | "
        f"Val Macro F1 {val_metrics['macro_f1']:.4f}"
    )

    if val_metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = (
            val_metrics["macro_f1"]
        )

        best_epoch = epoch + 1

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_v3_efficientnet_b0.pth"
        )

        print(
            f"  → New best model! "
            f"Macro F1 = {best_macro_f1:.4f}"
        )

    else:

        patience_counter += 1

        print(
            f"  → No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

best_model_path = (
    MODEL_DIR /
    "best_v3_efficientnet_b0.pth"
)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

model.eval()

print("\n")
print("=" * 70)
print("BEST V3 MODEL LOADED")
print("=" * 70)
print(
    f"Best Validation Macro F1: "
    f"{best_macro_f1:.4f}"
)
print(
    f"Best Stage 2 Epoch: "
    f"{best_epoch}"
)


# ============================================================
# 17. FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

all_test_labels = []
all_test_predictions = []
all_test_probabilities = []

test_loss = 0.0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        test_loss += (
            loss.item() *
            images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_test_labels.extend(
            labels.cpu().numpy()
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )

        all_test_probabilities.extend(
            probabilities.cpu().numpy()
        )


test_loss /= len(test_dataset)

test_metrics = calculate_metrics(
    all_test_labels,
    all_test_predictions
)


print(f"Test Loss: {test_loss:.4f}")
print(
    f"Test Accuracy: "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)
print(
    f"Macro Precision: "
    f"{test_metrics['macro_precision'] * 100:.2f}%"
)
print(
    f"Macro Recall: "
    f"{test_metrics['macro_recall'] * 100:.2f}%"
)
print(
    f"Macro F1: "
    f"{test_metrics['macro_f1'] * 100:.2f}%"
)
print(
    f"Weighted F1: "
    f"{test_metrics['weighted_f1'] * 100:.2f}%"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

class_names = [
    "KL0",
    "KL1",
    "KL2",
    "KL3",
    "KL4"
]

report = classification_report(
    all_test_labels,
    all_test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)


with open(
    RESULTS_DIR / "classification_report_v3.txt",
    "w"
) as f:

    f.write(report)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions
)

plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title(
    "OsteoSense V3 — Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "confusion_matrix_v3.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. TRAINING CURVES
# ============================================================

full_train_loss = (
    stage1_history["train_loss"] +
    stage2_history["train_loss"]
)

full_val_loss = (
    stage1_history["val_loss"] +
    stage2_history["val_loss"]
)

full_train_accuracy = (
    stage1_history["train_accuracy"] +
    stage2_history["train_accuracy"]
)

full_val_accuracy = (
    stage1_history["val_accuracy"] +
    stage2_history["val_accuracy"]
)

full_val_f1 = (
    stage1_history["val_macro_f1"] +
    stage2_history["val_macro_f1"]
)

epochs = range(
    1,
    len(full_train_loss) + 1
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    epochs,
    full_train_loss,
    label="Train Loss"
)

plt.plot(
    epochs,
    full_val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=STAGE1_EPOCHS,
    linestyle="--",
    label="Stage 1 → Stage 2"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "OsteoSense V3 — Loss Curves"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "loss_curve_v3.png",
    dpi=300
)

plt.show()


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    epochs,
    np.array(full_train_accuracy) * 100,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    np.array(full_val_accuracy) * 100,
    label="Validation Accuracy"
)

plt.axvline(
    x=STAGE1_EPOCHS,
    linestyle="--",
    label="Stage 1 → Stage 2"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "OsteoSense V3 — Accuracy Curves"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "accuracy_curve_v3.png",
    dpi=300
)

plt.show()


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    epochs,
    np.array(full_val_f1) * 100,
    label="Validation Macro F1"
)

plt.axvline(
    x=STAGE1_EPOCHS,
    linestyle="--",
    label="Stage 1 → Stage 2"
)

plt.axhline(
    y=best_macro_f1 * 100,
    linestyle=":",
    label=f"Best Macro F1 = {best_macro_f1 * 100:.2f}%"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1 (%)")

plt.title(
    "OsteoSense V3 — Validation Macro F1"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "macro_f1_curve_v3.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results = {
    "model": "EfficientNet-B0",
    "classes": class_names,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "stage1_epochs": STAGE1_EPOCHS,
    "stage2_epochs_max": STAGE2_EPOCHS,
    "focal_gamma": FOCAL_GAMMA,
    "optimizer": "AdamW",
    "backbone_lr": BACKBONE_LR,
    "classifier_lr": CLASSIFIER_LR,
    "weight_decay": WEIGHT_DECAY,
    "augmentation": [
        "CLAHE",
        "RandomRotation ±7 degrees",
        "RandomAffine translation 3%",
        "ColorJitter brightness 0.15",
        "ColorJitter contrast 0.15"
    ],
    "weighted_random_sampler": True,
    "early_stopping_patience": PATIENCE,
    "selection_metric": "Validation Macro F1",
    "best_validation_macro_f1": best_macro_f1,
    "test_loss": test_loss,
    "test_accuracy": test_metrics["accuracy"],
    "test_macro_precision": test_metrics["macro_precision"],
    "test_macro_recall": test_metrics["macro_recall"],
    "test_macro_f1": test_metrics["macro_f1"],
    "test_weighted_f1": test_metrics["weighted_f1"]
}

with open(
    RESULTS_DIR / "v3_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\n")
print("=" * 70)
print("V3 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Macro F1: "
    f"{best_macro_f1 * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_metrics['macro_f1'] * 100:.2f}%"
)

print(
    f"\nModel saved to:\n"
    f"{best_model_path}"
)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)

print("=" * 70)