import argparse
import csv
import random
import shutil
from pathlib import Path

import torch
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

ROOT = Path(__file__).resolve().parents[1]


class SquarePad:
    def __call__(self, image):
        # Make the image square with white padding so shapes are not stretched.
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, fill=255, padding_mode="constant")


def load_labels(weights_path: Path):
    # Read labels saved with the model weights (fallback to static labels).
    payload = torch.load(weights_path, map_location="cpu")
    labels = payload.get("labels")
    if labels:
        return labels
    from src.labels import LABELS

    return LABELS


def build_model(num_classes: int):
    # Use the same backbone as training.
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Export example predictions.")
    parser.add_argument("--weights", default=str(ROOT / "artifacts" / "model_weights.pt"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "crops" / "manifest.csv"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "examples"))
    parser.add_argument("--num-correct", type=int, default=6)
    parser.add_argument("--num-incorrect", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(weights_path)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = build_model(len(labels))
    payload = torch.load(weights_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Store examples we want to export.
    correct = []
    incorrect = []
    rows = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != "val":
                continue
            label = row["label"]
            if label not in label_to_idx:
                continue
            rows.append(row)

    random.seed(args.seed)
    random.shuffle(rows)

    for row in rows:
        # Stop when we already have enough examples.
        if len(correct) >= args.num_correct and len(incorrect) >= args.num_incorrect:
            break
        image_path = ROOT / "data" / "crops" / row["crop_path"]
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_label = labels[pred_idx]
        true_label = row["label"]
        score = float(probs[pred_idx])
        entry = {
            "image_path": image_path,
            "true": true_label,
            "pred": pred_label,
            "score": score,
            "source_image": row.get("source_image", ""),
            "bbox": row.get("bbox", ""),
        }
        if pred_label == true_label and len(correct) < args.num_correct:
            correct.append(entry)
        if pred_label != true_label and len(incorrect) < args.num_incorrect:
            incorrect.append(entry)

    results = [("correct", item) for item in correct] + [("incorrect", item) for item in incorrect]
    csv_path = out_dir / "examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "file", "true", "pred", "score", "source_image", "bbox"])
        for idx, (kind, item) in enumerate(results, start=1):
            name = f"{idx:03d}_{kind}_true-{item['true']}_pred-{item['pred']}_p{item['score']:.2f}.png"
            dest = out_dir / name
            shutil.copy2(item["image_path"], dest)
            writer.writerow(
                [
                    kind,
                    str(dest.relative_to(out_dir)),
                    item["true"],
                    item["pred"],
                    f"{item['score']:.4f}",
                    item["source_image"],
                    item["bbox"],
                ]
            )

    print(f"Wrote {len(results)} examples to {out_dir}")


if __name__ == "__main__":
    main()
