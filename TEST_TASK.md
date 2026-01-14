# Cabinet Classification - Test Task

## Overview
Build a cabinet-type classifier that can classify cabinet objects in architectural drawings. You may use an existing/pretrained model or build something from scratch; the key is that the final deliverable is a working classifier. The model should be able to classify cropped cabinet images into different types of casework cabinets from technical drawings and blueprints.

## Objective
Develop a classification model capable of categorizing cabinet images into the following cabinet types:
- **Base Cabinet - Open** (`lc:bcabo`)
- **Wall Cabinet - Open** (`lc:wcabo`)
- **Miscellaneous Cabinet - Insulated** (`lc:muscabinso`)
- **Wall Cabinet - Open Cubbie** (`lc:wcabcub`)
- **Base Cabinet - Open Cubbie** (`lc:bcabocub`)

## Dataset Description

### Dataset Overview
The dataset consists of architectural and engineering drawings from various educational and commercial construction projects. Images are primarily technical drawings, blueprints, and purchase sets containing detailed cabinet specifications.

### Dataset Statistics
- **Total Annotated Objects**: 2,107 cabinet instances
- **Number of Drawing Files**: 77 unique files
- **Number of Categories**: 5 cabinet types
- **Annotation Pages**: 232 pages with annotations

### Category Distribution
| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| `lc:bcabo` | 111 pages | 47.8% | Base Cabinet - Open |
| `lc:wcabo` | 49 pages | 21.1% | Wall Cabinet - Open |
| `lc:muscabinso` | 46 pages | 19.8% | Miscellaneous Cabinet - Insulated |
| `lc:wcabcub` | 22 pages | 9.5% | Wall Cabinet - Open Cubbie |
| `lc:bcabocub` | 4 pages | 1.7% | Base Cabinet - Open Cubbie |

### Dataset Format
- Annotations are provided in COCO JSON format
- Each project folder contains:
  - PDF drawings (source images)
  - `simple_annotations.json` file with bounding box annotations
  - Associated metadata

## Deliverables

1. **Explanation of Your Approach** (most important): Tell us about your process
   - **What did you try?** Share your experiments, even the ones that didn't work
   - **Why did you make those choices?** Explain your reasoning for architecture, hyperparameters, preprocessing, etc.
   - **What worked? What didn't?** Be honest about what performed well and what struggled
   - **What would you do differently with more time/resources?**
   - **What challenges did you face?** Technical issues, data problems, class imbalance, etc.

2. **Working Classifier**: A small, runnable project that includes both training and inference
   - **Training script**: trains a model and saves weights/artifacts
   - **Inference script**: loads saved weights and predicts a class + confidence for a given cabinet image (or a small folder of images)
   - Use any approach you want (pre-trained models, custom architectures, transfer learning, etc.)
   - All packaged and run in Docker (CLI args/examples are appreciated)
   - *Note: This doesn't need to be production-ready or highly accurate - we just want to see something runnable*

3. **Results**: Show us how your model performs
   - Basic metrics (accuracy, confusion matrix, or whatever you think is relevant)
   - A few example predictions (correct and incorrect)
   - Your interpretation of the results
   - Make a Loom recording where you briefly run the code and show how it works

## Questions?
If anything is unclear about the dataset or task, feel free to ask.

Good luck!
