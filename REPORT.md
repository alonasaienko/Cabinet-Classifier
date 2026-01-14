# Baseline Model Report

## Approach Summary
I built a simple baseline image classifier for cabinet crops using a pretrained ResNet-18. The goal was to get a working, reproducible pipeline with reasonable performance before experimenting with heavier models or custom architectures.

## What I Tried
- **Data prep**: Parsed `*_simple.json` annotations, cropped cabinet regions, and built a train/val manifest.
- **PNG vs PDF**: **Trained on both low-res PNGs and high-res PDFs. Surprisingly, the accuracy difference was negligible, but PDF rendering was kept for better pipeline control and coordinate precision.**
- **Sampling Ablation**: **Tested training without `WeightedRandomSampler`, which resulted in very poor performance (model collapsed to the majority class).**
- **PDF render fixes**: Added DPI=300 and scaled bbox coordinates to match the rendered page size; added small padding around crops.
- **Model**: ResNet-18 pretrained on ImageNet with a new FC head for 5 classes.
- **Preprocessing**: Square padding (white background), resize to 224x224, grayscale to 3 channels.
- **Augmentation Experiments**: Tested various strategies including RandomInvert and small-angle rotations, but converged on discrete 90° rotations and flips as they yielded the best stability.
- **Class imbalance handling**: WeightedRandomSampler in the training loader.

## Why These Choices
- **ResNet-18** is lightweight and stable for quick iteration.
- **SquarePad + resize** reduces distortion of thin blueprint lines.
- **Grayscale** simplifies line drawings and reduces color noise.
- **Discrete Rotation (90° only):** architectural symbols are strictly axis-aligned. Arbitrary rotations proved harmful, so I restricted augmentation to 90-degree increments.
- **WeightedRandomSampler**: Critical for convergence. Experiments confirmed that without this, the model completely fails to learn minority classes like `cubbies`.
- **PDF rendering + scaling** ensures crops match annotation coordinates, which are defined in the JSON image size.

## What Worked
- **WeightedRandomSampler**: This was the most effective component; removing it destroyed model performance.
- Training stabilized quickly; validation accuracy reached ~0.85–0.87 in later epochs.
- `lc:muscabinso` and `lc:wcabcub` were classified well on the small val set.
- The pipeline now produces reproducible crops and artifacts (loss curves, confusion matrix, weights).

## What Didn't Work Well
- `lc:wcabo` stayed weak and often confused with `lc:bcabo`. Overall accuracy was dominated by the largest class.
- High-Res Impact: Switching from PNG to high-res PDF did not drastically improve the `wcabo` vs `bcabo` confusion, suggesting the bottleneck is likely architectural rather than just resolution.
- Very small classes (e.g., `lc:bcabocub`) have too few samples in val, so their accuracy is not reliable.
- Early runs using raw PDF renders without scaling produced misaligned crops (fixed by scaling).
- Aggressive Augmentations: Arbitrary rotations (e.g., ±15°) caused aliasing on thin lines ("jagged edges"), confusing the model. RandomInvert also failed to improve performance and was discarded.

## What I Would Do With More Time
- Use **stratified split** and/or **cross-validation** to make evaluation more reliable.
- Try **class-balanced loss** (inverse frequency or focal loss) in addition to sampling.
- Collect more `wcabo`/`bcabocub` examples or apply targeted augmentation.
- Evaluate stronger backbones (EfficientNet, ConvNeXt) and add LR scheduling.
- Add automated error analysis: inspect top confusions and hard examples.

## Challenges Faced
- **Coordinate scaling**: bbox coordinates are in the JSON image resolution, which differs from PDF render resolution.
- **Class imbalance**: dominant `lc:bcabo` class biases the model.
- **Sparse classes**: some classes have very few samples, especially in validation.
