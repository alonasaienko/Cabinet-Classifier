import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.labels import LABELS
DATA_ROOT = ROOT / "data" / "crops"
IMAGES_ROOT = DATA_ROOT / "images"


def stable_split_key(value: str, val_ratio: float):
    # Use a hash so the split is stable across runs.
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def clamp_bbox(bbox, width, height, padding):
    # Clamp bbox to image bounds and add padding.
    x, y, w, h = bbox
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(width, int(round(x + w)))
    y1 = min(height, int(round(y + h)))
    if padding:
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(width, x1 + padding)
        y1 = min(height, y1 + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def iter_simple_json(root: Path):
    # Read only annotation JSON files, skip debug folders.
    for path in root.rglob("*_simple.json"):
        if "debug" in path.parts:
            continue
        yield path


def resolve_png_path(json_path: Path):
    # JSON filenames usually end with _simple.json.
    if json_path.name.endswith("_simple.json"):
        return json_path.with_name(json_path.name.replace("_simple.json", ".png"))
    return json_path.with_suffix(".png")


def resolve_pdf_path(json_path: Path, data: dict):
    # Try to find the original PDF for this page.
    file_info = data.get("file") or {}
    file_name = file_info.get("file_name")
    project_root = json_path.parent.parent
    if file_name:
        candidate = project_root / file_name
        if candidate.exists():
            return candidate
    pdfs = list(project_root.glob("*.pdf"))
    if not pdfs:
        return None
    for pdf in pdfs:
        if pdf.name.endswith("_original.pdf"):
            return pdf
    return pdfs[0]


def page_number_from_data(json_path: Path, data: dict):
    # Page number is stored in JSON, fallback to filename suffix.
    images = data.get("images") or []
    if images:
        page_number = images[0].get("page_number")
        if isinstance(page_number, int):
            return page_number
    stem = json_path.stem
    if "_" in stem:
        tail = stem.split("_")[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def prepare_dataset(
    source_root: Path,
    val_ratio: float,
    clean: bool,
    source_format: str,
    dpi: int,
    padding: int):
    # Main pipeline: load pages, crop boxes, and write manifest.
    if clean and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    entries = []

    for json_path in iter_simple_json(source_root):
        payload = json_path.read_text(encoding="utf-8")
        try:
            data = __import__("json").loads(payload)
        except ValueError:
            continue

        annotations = data.get("annotations") or []
        if not annotations:
            continue

        if source_format == "pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError:
                raise SystemExit("pdf2image is required for --source-format pdf.")

            pdf_path = resolve_pdf_path(json_path, data)
            if pdf_path is None:
                continue
            page_number = page_number_from_data(json_path, data)
            split_key = f"{pdf_path.relative_to(source_root)}#page={page_number}"
            try:
                images = convert_from_path(
                    str(pdf_path),
                    dpi=dpi,
                    first_page=page_number + 1,
                    last_page=page_number + 1,
                )
            except Exception as exc:
                print(f"Skipping {pdf_path} page {page_number}: {exc}")
                continue
            if not images:
                continue
            image = images[0].convert("RGB")
            source_image = f"{pdf_path.relative_to(source_root)}#page={page_number}"
            stem = f"{pdf_path.stem}_p{page_number:05d}"
            ref_images = data.get("images") or []
            if ref_images:
                ref_width = ref_images[0].get("width")
                ref_height = ref_images[0].get("height")
            else:
                ref_width = None
                ref_height = None
        else:
            image_path = resolve_png_path(json_path)
            if not image_path.exists():
                continue
            split_key = str(image_path.relative_to(source_root))
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            source_image = str(image_path.relative_to(source_root))
            stem = image_path.stem
            ref_width = None
            ref_height = None

        split = stable_split_key(split_key, val_ratio)

        width, height = image.size
        scale_x = 1.0
        scale_y = 1.0
        if ref_width and ref_height:
            if ref_width > 0 and ref_height > 0:
                scale_x = width / ref_width
                scale_y = height / ref_height
        for idx, ann in enumerate(annotations):
            label = ann.get("label") or ann.get("object_text")
            if label not in LABELS:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            scaled_bbox = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            ]
            clipped = clamp_bbox(scaled_bbox, width, height, padding)
            if clipped is None:
                continue
            x0, y0, x1, y1 = clipped
            crop = image.crop((x0, y0, x1, y1))

            rel_dir = Path(split) / label
            out_dir = IMAGES_ROOT / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{stem}_ann{idx:04d}.png"
            out_path = out_dir / out_name
            crop.save(out_path)

            entries.append(
                {
                    "crop_path": str(Path("images") / rel_dir / out_name),
                    "label": label,
                    "split": split,
                    "source_image": source_image,
                    "bbox": f"{x0},{y0},{x1 - x0},{y1 - y0}",
                    "abs_path": out_path,
                }
            )

    # Ensure at least one example per class in validation.
    missing_in_val = {label for label in LABELS}
    for entry in entries:
        if entry["split"] == "val":
            missing_in_val.discard(entry["label"])

    if missing_in_val:
        for label in sorted(missing_in_val):
            candidate = next(
                (entry for entry in entries if entry["label"] == label and entry["split"] == "train"),
                None,
            )
            if candidate is None:
                continue
            src_path = candidate["abs_path"]
            rel_dir = Path("val") / label
            dest_dir = IMAGES_ROOT / rel_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name
            shutil.move(str(src_path), str(dest_path))
            candidate["split"] = "val"
            candidate["crop_path"] = str(Path("images") / rel_dir / src_path.name)
            candidate["abs_path"] = dest_path

    manifest_path = DATA_ROOT / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "crop_path",
                "label",
                "split",
                "source_image",
                "bbox",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "crop_path": entry["crop_path"],
                    "label": entry["label"],
                    "split": entry["split"],
                    "source_image": entry["source_image"],
                    "bbox": entry["bbox"],
                }
            )

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Prepare cabinet crops dataset.")
    parser.add_argument(
        "--source",
        default=str(ROOT / "annotated_pdfs_and_data"),
        help="Path to annotated_pdfs_and_data folder.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--source-format",
        choices=("png", "pdf"),
        default="png",
        help="Read page images from PNGs or render them from PDFs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF rendering.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Padding in pixels around each crop.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing prepared dataset before regenerating.",
    )
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    if not source_root.exists():
        raise SystemExit(f"Source folder not found: {source_root}")

    manifest_path = prepare_dataset(
        source_root,
        args.val_ratio,
        args.clean,
        args.source_format,
        args.dpi,
        args.padding,
    )
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
