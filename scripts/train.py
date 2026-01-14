import csv
import sys
from pathlib import Path

import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
ARTIFACTS_DIR = ROOT / "artifacts1"

from src.labels import LABELS, LABEL_TO_INDEX

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

data_root = ROOT / "data" / "crops"
manifest_path = data_root / "manifest.csv"


class CropsDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, split: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if manifest.exists():
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row["split"] != split:
                        continue
                    label = row["label"]
                    if label not in LABEL_TO_INDEX:
                        continue
                    rel_path = Path(row["crop_path"])
                    self.samples.append((root / rel_path, LABEL_TO_INDEX[label]))
        else:
            split_dir = root / split
            image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            for label in LABELS:
                label_dir = split_dir / label
                if not label_dir.exists():
                    continue
                for path in label_dir.rglob("*"):
                    if path.is_file() and path.suffix.lower() in image_exts:
                        self.samples.append((path, LABEL_TO_INDEX[label]))

        if not self.samples:
            raise ValueError(
                f"No samples found for split='{split}'. Expected {manifest} or "
                f"folders in {root / split}."
            )
        
        # Save targets for sampler.
        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

model = models.resnet18(pretrained=True)
num_classes = len(LABELS)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        # Make image square with white padding so aspect ratio is kept.
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, fill=255, padding_mode='constant')

transform_train = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(degrees=(-90, 90)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_val = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CropsDataset(data_root, manifest_path, "train", transform=transform_train)
val_dataset = CropsDataset(data_root, manifest_path, "val", transform=transform_val)

targets = train_dataset.targets 
class_counts = np.bincount(targets)
class_weights = 1. / np.sqrt(class_counts)
class_weights = torch.FloatTensor(class_weights)
samples_weights = class_weights[targets]
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(train_dataset), replacement=True)

# Use sampler instead of shuffle for class balance.
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Determine whether to use GPU (if available) or CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Keep history for plots.
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    final_confusion = None

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for t, p in zip(labels.tolist(), preds.tolist()):
                    confusion[t][p] += 1

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.float() / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc.item())
        history["val_acc"].append(val_acc.item())
        final_confusion = confusion

        print(
            f"Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, "
            f"train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )
        print("Confusion matrix (rows=true, cols=pred):")
        print("labels:", LABELS)
        for row in confusion:
            print(row)

    return history, final_confusion

model = model.to(device)
history, final_confusion = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Save weights for inference.
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
weights_path = ARTIFACTS_DIR / "model_weights.pt"
torch.save({"state_dict": model.state_dict(), "labels": LABELS}, weights_path)

# Save metrics to CSV.
history_path = ARTIFACTS_DIR / "train_history.csv"
with history_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
    for idx, (tl, vl, ta, va) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["train_acc"], history["val_acc"]),
        start=1,
    ):
        writer.writerow([idx, f"{tl:.6f}", f"{vl:.6f}", f"{ta:.6f}", f"{va:.6f}"])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history["train_loss"], label="train loss")
ax.plot(history["val_loss"], label="val loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "loss_curve.png", dpi=150)
plt.close(fig)

# Save confusion matrix image for the last epoch.
if final_confusion:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(final_confusion, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, final_confusion[i][j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_loader, device):
    classes = LABELS
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for label, prediction in zip(labels, preds):
                classname = classes[label.item()]
                if label == prediction:
                    correct_pred[classname] += 1
                total_pred[classname] += 1

    accuracy_per_class = {
        classname: correct_pred[classname] / total_pred[classname] if total_pred[classname] > 0 else 0
        for classname in classes
    }

    overall_accuracy = accuracy_score(all_labels, all_preds)

    print("Accuracy per class:")
    for classname, accuracy in accuracy_per_class.items():
        print(f"{classname}: {accuracy:.4f}")

    print()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")



evaluate_model(model, val_loader, device)
