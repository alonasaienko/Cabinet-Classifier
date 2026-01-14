# Architectural Symbol Classifier

A computer vision pipeline to classify cabinet symbols from blueprint PDFs.
Built as a test assignment.

## Results (Baseline)
- Overall accuracy: ~0.85-0.87 on validation.
- Model: ResNet-18 (ImageNet pretrained) with a custom 5-class head.
- Data: PDF-rendered crops with bbox scaling and padding.

## Prerequisites
- Docker + Docker Compose.
- Source PDFs/annotations in `./annotated_pdfs_and_data`.

## Quick Start (GPU)
Build the GPU image and run the main scripts:

```bash
sudo docker compose build
sudo docker compose run --rm classifier \
  python scripts/prepare_dataset.py --clean --source-format pdf
sudo docker compose run --rm classifier \
  python scripts/train.py
sudo docker compose run --rm classifier \
  python scripts/infer_examples.py --num-correct 6 --num-incorrect 6
```

## Quick Start (CPU)
If you do not have GPU support, use the CPU image:

```bash
docker build -f Dockerfile.cpu -t cabinet-classifier-cpu .
docker run --rm -it \
  -v "$(pwd)/annotated_pdfs_and_data:/app/annotated_pdfs_and_data" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  cabinet-classifier-cpu \
  python scripts/prepare_dataset.py --clean --source-format pdf
```

## Scripts
- `scripts/prepare_dataset.py`: builds `data/crops` and `manifest.csv`.
  - Example: `python scripts/prepare_dataset.py --clean --source-format pdf --dpi 300 --padding 12`
- `scripts/train.py`: trains the model and saves artifacts.
  - Artifacts: `artifacts/model_weights.pt`, `artifacts/loss_curve.png`,
    `artifacts/confusion_matrix.png`, `artifacts/train_history.csv`.
- `scripts/infer_examples.py --num-correct 6 --num-incorrect 6`: exports correct/incorrect examples.
  - Output: `reports/examples/` + `reports/examples/examples.csv`.

## Notes
- GPU compose uses `Dockerfile.gpu`.
- CPU image uses `Dockerfile.cpu`.
