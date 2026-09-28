import os
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.models import resnet18,ResNet18_Weights
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n==============================================")
print("           OsteoSense Training")
print("==============================================")
print(f"Device: {device}")

if device.type=="cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

DATA_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\DATA")
TRAIN_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\DATA\train")
VAL_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\DATA\val")
TEST_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\DATA\test")
MODEL_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\MODEL")
RESULTS_DIR=Path(r"C:\Users\kinnu\OneDrive\Desktop\SIH OA\X-RAY MODEL\RESULTS")

NUM_CLASSES=5
BATCH_SIZE=16
NUM_EPOCHS=100
LEARNING_RATE=0.0001
WEIGHT_DECAY=1e-4
IMAGE_SIZE=224
NUM_WORKERS=0
PIN_MEMORY=False

train_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_test_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

print("\nLoading Datasets...........")

train_dataset=datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset=datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset=datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)

print("\nClass mapping : ")
print(train_dataset.class_to_idx)

for index,class_name in enumerate(train_dataset.classes):
    print(f"{index} -> KL GRADE {class_name}")

print("\nDataset sizes: ")
print(f"Training   : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Testing    : {len(test_dataset)}")

print("\nCalculating class weights......")

train_targets=np.array(train_dataset.targets)

class_counts=np.bincount(
    train_targets,
    minlength=NUM_CLASSES
)

print("\nTraining class distribution: ")

for class_index,count in enumerate(class_counts):
    print(f"KL GRADE {class_index}: {count} images")

class_weights=(
    len(train_targets)/
    (NUM_CLASSES*class_counts)
)

class_weights=torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass weights: ")

for i,weight in enumerate(class_weights):
    print(f"KL GRADE {i} : {weight:4f}")

class_weights=class_weights.to(device)

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

print("\nLoading pretrained ResNet-18......")

weights=ResNet18_Weights.DEFAULT

model=resnet18(
    weights=weights
)

num_features=model.fc.in_features

model.fc=nn.Linear(
    num_features,
    NUM_CLASSES
)

model=model.to(device)

print("\nModel: ")
print(model.fc)

criterion=nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer=optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler=optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

def train_one_epoch(model,loader,criterion,optimizer):

    model.train()

    running_loss=0.0
    correct=0
    total=0

    for images,labels in loader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss+=(
            loss.item()*images.size(0)
        )

        predicted=torch.argmax(
            outputs,
            dim=1
        )

        total+=labels.size(0)

        correct+=torch.eq(
            predicted,
            labels
        ).sum().item()

    epoch_loss=running_loss/total

    epoch_accuracy=correct/total

    return epoch_loss,epoch_accuracy


def validate(model,loader,criterion):

    model.eval()

    running_loss=0.0

    all_predictions=[]
    all_labels=[]

    with torch.no_grad():

        for images,labels in loader:

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)

            loss=criterion(
                outputs,
                labels
            )

            running_loss+=(
                loss.item()*images.size(0)
            )

            predictions=torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    total=len(all_labels)

    epoch_loss=running_loss/total

    accuracy=accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1=f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return(
        epoch_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


print("\n==============================================")
print("Starting training...")
print("==============================================\n")

best_val_f1=0.0

history={
    "train_loss":[],
    "train_accuracy":[],
    "val_loss":[],
    "val_accuracy":[],
    "val_f1":[]
}

start_time=time.time()

for epoch in range(NUM_EPOCHS):

    epoch_start=time.time()

    print(
        f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
    )

    train_loss,train_accuracy=train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss,val_accuracy,val_f1,_,_=validate(
        model,
        val_loader,
        criterion
    )

    scheduler.step(val_loss)

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["val_f1"].append(
        val_f1
    )

    epoch_time=time.time()-epoch_start

    current_lr=optimizer.param_groups[0]["lr"]

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Val Macro F1: {val_f1:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )

    print(
        f"Epoch Time: {epoch_time/60:.2f} minutes"
    )

    if val_f1>best_val_f1:

        best_val_f1=val_f1

        checkpoint_path=(
            MODEL_DIR/
            "osteosense_resnet18_best.pth"
        )

        torch.save(
            {
                "model_state_dict":model.state_dict(),
                "class_to_idx":train_dataset.class_to_idx,
                "best_val_f1":best_val_f1,
                "epoch":epoch+1
            },
            checkpoint_path
        )

        print("\n*** BEST MODEL SAVED ***")


total_time=time.time()-start_time

print("\n==============================================")
print("Training complete")
print("==============================================")

print(
    f"Total training time: "
    f"{total_time/3600:.2f} hours"
)

print(
    f"Best validation Macro F1: "
    f"{best_val_f1:.4f}"
)

history_path=RESULTS_DIR/"training_history.json"

with open(history_path,"w") as f:
    json.dump(
        history,
        f,
        indent=4
    )

plt.figure(figsize=(8,6))

plt.plot(
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "OsteoSense - Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"loss_curve.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(8,6))

plt.plot(
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "OsteoSense - Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"accuracy_curve.png",
    dpi=300
)

plt.close()

print("\nLoading best model...")

checkpoint=torch.load(
    MODEL_DIR/"osteosense_resnet18_best.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch: {checkpoint['epoch']}"
)

print(
    f"Best validation Macro F1: "
    f"{checkpoint['best_val_f1']:.4f}"
)

print("\n==============================================")
print("Final Test Evaluation")
print("==============================================")

test_loss,test_accuracy,test_f1,test_labels,test_predictions=validate(
    model,
    test_loader,
    criterion
)

test_precision=precision_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_recall=recall_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

print(
    f"\nTest Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Precision: {test_precision:.4f}"
)

print(
    f"Test Recall: {test_recall:.4f}"
)

print(
    f"Test Macro F1: {test_f1:.4f}"
)

report=classification_report(
    test_labels,
    test_predictions,
    target_names=[
        "KL GRADE 0",
        "KL GRADE 1",
        "KL GRADE 2",
        "KL GRADE 3",
        "KL GRADE 4"
    ],
    zero_division=0
)

print("\nClassification Report:")

print(report)

with open(
    RESULTS_DIR/"classification_report.txt",
    "w"
) as f:

    f.write(report)

cm=confusion_matrix(
    test_labels,
    test_predictions
)

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "KL 0",
        "KL 1",
        "KL 2",
        "KL 3",
        "KL 4"
    ]
)

fig,ax=plt.subplots(
    figsize=(8,8)
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "OsteoSense - KL Grade Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"confusion_matrix.png",
    dpi=300
)

plt.close()

final_metrics={
    "test_accuracy":float(test_accuracy),
    "test_precision_macro":float(test_precision),
    "test_recall_macro":float(test_recall),
    "test_f1_macro":float(test_f1),
    "best_validation_f1":float(best_val_f1),
    "best_epoch":int(checkpoint["epoch"])
}

with open(
    RESULTS_DIR/"final_metrics.json",
    "w"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )

print("\n==============================================")
print("OsteoSense model training pipeline finished.")
print("==============================================")

print("\nGenerated files:")

print("Model:")
print(
    "  models/osteosense_resnet18_best.pth"
)

print("\nResults:")

print(
    "  results/training_history.json"
)

print(
    "  results/loss_curve.png"
)

print(
    "  results/accuracy_curve.png"
)

print(
    "  results/classification_report.txt"
)

print(
    "  results/confusion_matrix.png"
)

print(
    "  results/final_metrics.json"
)

print("\n")