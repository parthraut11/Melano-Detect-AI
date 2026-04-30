"""
Colab-ready training script for HAM10000 skin lesion classification.

How to use in Google Colab:

1. Upload this file to Colab.
2. Upload your Kaggle API key file `kaggle.json` when prompted.
3. Run:

    !python train_skin_cancer_colab.py

Outputs:
- /content/skin_cancer_outputs/best_model.pth
- /content/skin_cancer_outputs/class_names.json
- /content/skin_cancer_outputs/training_summary.json

Optional download in Colab:

    from google.colab import files
    files.download('/content/skin_cancer_outputs/best_model.pth')
    files.download('/content/skin_cancer_outputs/class_names.json')
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
NUM_CLASSES = 7
OUTPUT_DIR = Path("/content/skin_cancer_outputs")
DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"

# Dataset-wide normalization values reused from the notebooks.
NORM_MEAN = [0.7630392, 0.5456477, 0.57004845]
NORM_STD = [0.1409286, 0.15261266, 0.16997074]


LESION_TYPE_DICT = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_package(package_name: str, import_name: str | None = None) -> None:
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        run_command([sys.executable, "-m", "pip", "install", "-q", package_name])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except ImportError:
        return False


def prepare_kaggle_credentials() -> None:
    if Path("/root/.kaggle/kaggle.json").exists():
        return

    if not in_colab():
        raise RuntimeError(
            "kaggle.json was not found. This script is meant to be run in Colab "
            "or on a machine with Kaggle credentials already configured."
        )

    from google.colab import files  # type: ignore

    print("Upload your Kaggle API key file: kaggle.json")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise RuntimeError("Upload cancelled or kaggle.json not provided.")

    kaggle_dir = Path("/root/.kaggle")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = kaggle_dir / "kaggle.json"
    with target.open("wb") as handle:
        handle.write(uploaded["kaggle.json"])
    os.chmod(target, 0o600)


def download_dataset() -> Path:
    ensure_package("kaggle")
    prepare_kaggle_credentials()

    dataset_dir = Path("/content/ham10000")
    if dataset_dir.exists() and (dataset_dir / "HAM10000_metadata.csv").exists():
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(dataset_dir),
            "--unzip",
        ]
    )
    return dataset_dir


def build_image_index(dataset_dir: Path) -> dict[str, str]:
    image_paths = list(dataset_dir.glob("**/*.jpg"))
    return {path.stem: str(path) for path in image_paths}


def prepare_dataframe(dataset_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    metadata_path = dataset_dir / "HAM10000_metadata.csv"
    df = pd.read_csv(metadata_path)

    image_index = build_image_index(dataset_dir)
    df["path"] = df["image_id"].map(image_index.get)
    df = df[df["path"].notna()].copy()

    df["cell_type"] = df["dx"].map(LESION_TYPE_DICT)
    class_names = sorted(df["cell_type"].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    df["cell_type_idx"] = df["cell_type"].map(class_to_idx)

    lesion_counts = df.groupby("lesion_id")["image_id"].count()
    unduplicated_ids = set(lesion_counts[lesion_counts == 1].index)
    df["duplicates"] = df["lesion_id"].apply(
        lambda lesion_id: "unduplicated" if lesion_id in unduplicated_ids else "duplicated"
    )

    df_undup = df[df["duplicates"] == "unduplicated"].copy()
    _, df_val = train_test_split(
        df_undup,
        test_size=0.2,
        random_state=101,
        stratify=df_undup["cell_type_idx"],
    )

    train_mask = ~df["image_id"].isin(df_val["image_id"])
    df_train = df[train_mask].copy()

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), class_names


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["cell_type_idx"])
        return image, label


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

    return train_transform, val_transform


def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@dataclass
class EpochMetrics:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_balanced_acc: float


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    loss_value = running_loss / len(loader.dataset)
    accuracy = float(np.mean(np.array(predictions) == np.array(targets)))
    balanced_acc = balanced_accuracy_score(targets, predictions)
    return loss_value, accuracy, balanced_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress.set_description(f"train loss={loss.item():.4f}")

    return running_loss / total, correct / total


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = download_dataset()
    df_train, df_val, class_names = prepare_dataframe(dataset_dir)

    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Classes: {class_names}")

    train_transform, val_transform = build_transforms()
    train_dataset = HAM10000Dataset(df_train, train_transform)
    val_dataset = HAM10000Dataset(df_val, val_transform)

    train_sampler = make_sampler(df_train["cell_type_idx"].tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    class_counts = df_train["cell_type_idx"].value_counts().sort_index().tolist()
    class_weights = torch.tensor(
        [len(df_train) / (NUM_CLASSES * count) for count in class_counts],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_score = -1.0
    history: list[dict[str, float]] = []
    best_model_path = OUTPUT_DIR / "best_model.pth"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_balanced_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_balanced_acc)

        metrics = EpochMetrics(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_balanced_acc=val_balanced_acc,
        )
        history.append(metrics.__dict__)

        print(
            f"train_loss={metrics.train_loss:.4f} "
            f"train_acc={metrics.train_acc:.4f} "
            f"val_loss={metrics.val_loss:.4f} "
            f"val_acc={metrics.val_acc:.4f} "
            f"val_bal_acc={metrics.val_balanced_acc:.4f}"
        )

        if val_balanced_acc > best_score:
            best_score = val_balanced_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    class_names_path = OUTPUT_DIR / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)

    summary = {
        "best_val_balanced_accuracy": best_score,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "class_names": class_names,
        "history": history,
    }
    summary_path = OUTPUT_DIR / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTraining complete.")
    print(f"Best model: {best_model_path}")
    print(f"Class names: {class_names_path}")
    print(f"Summary: {summary_path}")

    if in_colab():
        print(
            "\nDownload in Colab with:\n"
            "from google.colab import files\n"
            "files.download('/content/skin_cancer_outputs/best_model.pth')\n"
            "files.download('/content/skin_cancer_outputs/class_names.json')"
        )


if __name__ == "__main__":
    main()
