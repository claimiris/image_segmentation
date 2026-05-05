# Image Segmentation: TransUNet vs UNet++ on ISIC 2018

A unified pipeline for comparative medical image segmentation on the **ISIC 2018** (Skin Lesion Analysis) dataset, evaluating and contrasting the performance of **TransUNet** and **UNet++** architectures.

This repository integrates formerly separate model implementations into a single, clean, and modular codebase. It incorporates performance optimizations suitable for environments like Google Colab and offers comprehensive tools for training, evaluating, and visualizing model predictions.

## Key Features

- **Advanced Architectures**: Full implementations of **UNet++** (nested dense architectures) and **TransUNet** (CNN-Transformer hybrid).
- **Unified Pipeline**: A central structure for dataset loading, augmentation, training loops, and validation to ensure fair comparisons.
- **Performance Optimizations**: 
  - Substantial improvements tailored for efficiency, including `torch.compile` support.
  - Reduced Transformer depth settings for TransUNet to run effectively under memory constraints.
  - Efficient DataLoader implementations ensuring steady GPU saturation.
- **Robust Metrics & Losses**: Custom Dice Loss and Binary Cross Entropy (BCE) loss strategies combined with thorough evaluation metrics (IoU, Dice Coefficient).

## Project Structure

```text
.
├── models/
│   ├── transunet.py         # TransUNet architecture definition
│   └── unetplusplus.py      # UNet++ architecture definition
├── results/                 # Output directory for logs, metrics, and visualization plots
├── src/
│   ├── dataset.py           # Dataset loading and transforms (ISIC 2018 processing)
│   ├── trainer.py           # Core training loop, validation, and logging logic
│   ├── losses.py            # Definition of Dice and combined BCE-Dice losses
│   └── metrics.py           # Calculation of evaluation metrics (IoU, Dice)
├── .gitignore               # Git ignore file (excludes data/ and checkpoints)
├── README.md                # Project documentation
└── combined_run.ipynb       # Main entry point for training, evaluation, and visualization

# Note: The data/ directory is gitignored but must be created locally.
data/                        # Directory for ISIC 2018 training/validation datasets
```

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-capable GPU (or environment like Google Colab). Key dependencies include:
- `torch` & `torchvision`
- `numpy`
- `matplotlib`
- `PIL` (Pillow)
- `tqdm`
- `scikit-learn`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd image_segmentation
   ```
2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy matplotlib pillow tqdm scikit-learn
   ```

### Dataset Preparation

1. Download the **ISIC 2018: Task 1 (Lesion Boundary Segmentation)** dataset.
2. Extract the images and their corresponding mask ground truths into the `data/` directory. Ensure the paths map correctly within `src/dataset.py`.

## Usage

The main interactive workflow is managed inside `combined_run.ipynb`.

1. Launch Jupyter Notebook or upload the project to Google Colab.
2. Open `combined_run.ipynb`.
3. Follow the sequence of cells to:
   - Initialize the ISIC dataloaders.
   - Instantiate the UNet++ and TransUNet models.
   - Train the models (metrics are saved incrementally).
   - Evaluate model performances with visual side-by-side overlays of predictions vs. ground truth.

### Artifacts

During training, best-performing models will be automatically saved in the root directory (e.g., `TransUNet_best.pth`, `UNetPP_best.pth`), while visual graphs of the loss and IoU curves will be saved under the `results/` folder.

## Acknowledgements

- Based on the foundational papers for **[UNet++](https://arxiv.org/abs/1807.10165)** and **[TransUNet](https://arxiv.org/abs/2102.04306)**.
- Dataset provided by the [ISIC Archive](https://www.isic-archive.com/) (2018 Challenge).