# NAT-SLN-Metastasis-Prediction

**Research code for predicting sentinel lymph node (SLN) metastasis after neoadjuvant therapy (NAT) using breast lesion ultrasound, multimodal SLN ultrasound, and clinical variables.**

This repository contains independently trained deep learning models and R scripts for subsequent probability-level fusion, clinical modeling, and nomogram construction. **No patient-level clinical data, ultrasound images, ROI annotations, or individual prediction files are publicly distributed.**

## Repository structure

```text
NAT-SLN-Metastasis-Prediction/
├── training/
│   ├── breast_lesion_model.py   # Independent breast lesion model
│   ├── sln_model.py             # Independent SLN model
│   └── model_core.py            # Preprocessing, architectures, training, evaluation
├── nomogram/
│   ├── common.R                 # Shared R functions
│   ├── clinical_nomogram.R
│   ├── clinical_lesion_nomogram.R
│   ├── clinical_sln_nomogram.R
│   ├── lesion_sln_nomogram.R
│   ├── comprehensive_nomogram.R
│   └── README.md
├── data/
│   └── README.md               # Input-format documentation only
├── .gitignore
├── requirements.txt
└── README.md
```

Preprocessing and evaluation are implemented within the training scripts; no separate `scripts/`, `examples/`, or `evaluation/` directory is required.

## Modeling workflow

### 1. Breast lesion imaging model

Grayscale ultrasound images acquired **before and after NAT** are processed using a **shared-weight ImageNet-pretrained ResNet18**. A temporal attention module combines the two 512-dimensional representations, followed by a binary classification head. The output is the breast lesion model's predicted probability (`DL_lesion`).

### 2. SLN imaging model

SLN **grayscale**, **percutaneous contrast-enhanced ultrasound (PCEUS)**, and **intravenous contrast-enhanced ultrasound (IVCEUS)** images are processed by **three separate ImageNet-pretrained ResNet18 backbones**. A **four-head self-attention** module integrates the three modality representations. Mean pooling and a binary classification head yield the SLN model's predicted probability (`DL_SLN`).

### 3. Probability-level fusion and nomograms

The two imaging models are trained **independently**. There is **no feature concatenation or joint breast lesion–SLN neural-network classifier**. The downstream R workflow uses the two predicted probabilities in logistic regression; `nomogram/comprehensive_nomogram.R` converts them to logits, fits a logistic regression imaging signature on the training set, and combines that signature with five clinical predictors in a comprehensive nomogram.

The R scripts additionally implement clinical-only, clinical-plus-lesion, clinical-plus-SLN, and lesion-plus-SLN models. See [`nomogram/README.md`](nomogram/README.md) for the exact analysis entry points.

## Software environment

The reported deep learning environment was:

| Component | Version |
| --- | --- |
| Python | 3.9.23 |
| PyCharm (development IDE) | 2025.1 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| OpenCV | 4.12.0 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| scikit-learn | 1.6.1 |
| PyTorch / torchvision CUDA build | 12.8 |

PyTorch and torchvision were compiled with CUDA 12.8 support for GPU acceleration. PyCharm is an optional development environment, not a runtime dependency. The Python scripts also import Pillow, matplotlib, tqdm, and openpyxl; their exact historical versions were not supplied and are not claimed here.

The reported statistical analysis environment was:

| Component | Version |
| --- | --- |
| R | 4.5.0 |
| rms | 8.0-0 |
| readr | 2.1.5 |

**Evaluation responsibilities:** Python uses scikit-learn and matplotlib for deep learning ROC/AUC analysis and training curves. The R `common.R` script uses `pROC` to compute AUC for nomogram predictions; it uses `rms` for nomograms and bootstrap calibration. The R scripts were reported by the study team to run successfully and agree with the final study analysis.

### Python installation

Create an environment with Python 3.9.23. Install PyTorch and torchvision builds appropriate for your operating system, GPU, and CUDA runtime; for the reported experiment, the builds were `2.8.0+cu128` and `0.23.0+cu128`. Refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for platform-specific commands.

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

**Note:** `requirements.txt` lists package names, not a fully pinned environment lockfile. The version table above documents the versions reported for the study; reproducibility also depends on platform and hardware compatibility. Pretrained ImageNet weights may need to be downloaded on the first run.

### R installation

Install R 4.5.0 and the packages `rms`, `readr`, and `pROC` in a compatible R environment. Package versions not explicitly recorded should not be interpreted as exact reproductions of the original installation.

## Private input data (not provided)

### Python model inputs

The training scripts expect a **local Excel metadata file** containing at least `case_number`, `number`, `label` (0/1), and `dataset` (`train`/`test`). For safe patient-level partitioning, supply a separate, verified patient identifier and pass it using `--group-column`. The original `number` field is retained for compatibility **but must not be assumed to identify a unique patient**.

Each local case directory must contain the relevant ultrasound modality folders. The JSON loader expects base64-encoded `imageData` and ROI polygon points in `shapes[*].points` or top-level `points`.

| Input | Public-facing default directory |
| --- | --- |
| Pre-NAT breast lesion grayscale | `lesion_pre_nat` |
| Post-NAT breast lesion grayscale | `lesion_post_nat` |
| SLN grayscale | `sln_grayscale` |
| SLN PCEUS | `sln_pceus` |
| SLN IVCEUS | `sln_ivceus` |

Use the corresponding `--*-folder` command-line options if your **private** directory names differ. Do not publish original case names or patient identifiers to demonstrate folder structure.

### R model inputs

`nomogram/common.R` reads two private files, `data/processed/train_all.csv` and `data/processed/test_all.csv`. Depending on the analysis, the expected fields include `patient_id`, `sln_id`, `label`, `Ki_67`, `SLN_diameter`, `HER_2`, `cortex_thickness`, `N_grade_pred`, `DL_lesion`, and `DL_SLN`. These are **column names only**, not released observations. The scripts check for duplicate SLN identifiers and overlap in patient identifiers across training and test sets.

The imaging probabilities supplied to the R scripts must correspond to the **same SLN-level observations** and must be aligned using a verified identifier. To avoid optimistic performance estimates, probability-fusion models should be fitted with appropriately held-out or out-of-fold base-model predictions.

## Image preprocessing

The existing Python code performs the following operations:

1. Decode the image from JSON using OpenCV and convert BGR to RGB.
2. Apply the first available ROI polygon as a mask, set pixels outside the polygon to zero, and crop to the polygon bounding rectangle. If no ROI is found, the code falls back to the full image.
3. Resize to `224 × 224` pixels (without preserving the original aspect ratio).
4. During training, apply random horizontal and vertical flips, rotation up to 15 degrees, and brightness/contrast jitter (`0.2`). Augmentations are sampled independently for each input image.
5. Convert to tensors and normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

No additional CLAHE, speckle denoising, image registration, or synchronized multimodal geometric augmentation is implemented in the supplied training code.

## Training and evaluation

| Setting | Implementation |
| --- | --- |
| Maximum epochs | 20 |
| Batch size | 16 |
| Random seed | 27 |
| Learning rate | `3e-5` |
| Weight decay | `5e-4` |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Loss | BCEWithLogitsLoss with training-derived positive-class weight |
| Internal validation | Single patient-grouped stratified split, nominally 20% of development data |
| Checkpoint selection | Best validation AUC exponential moving average (EMA) |
| Early stopping | Default patience = 3; `--patience 0` disables it |

The implementation is **not five-fold cross-validation**. The independent test set is not used by the checkpoint-selection or early-stopping rule; however, the current script evaluates test metrics during training, which can create a risk of indirect test-set influence if results are used to guide further development. Reported training/development metrics should not be confused with fully out-of-sample estimates.

### Running the imaging models

From the repository root, with private paths and a validated patient grouping field:

```bash
python training/breast_lesion_model.py --excel-path /private/metadata.xlsx --image-root /private/cases --group-column patient_group_id --patience 3
python training/sln_model.py --excel-path /private/metadata.xlsx --image-root /private/cases --group-column patient_group_id --patience 3
```

The above paths and `patient_group_id` are **placeholders**, not released data. The grouping column must exist in your local metadata. The code also deduplicates `case_number` values; confirm this does not discard valid SLN observations before training.

Training outputs, model weights, curves, and patient-level prediction CSV files are written locally under `train_test_outputs/` and must remain private.

### Running the R analyses

From the repository root, after preparing the private inputs:

```bash
Rscript nomogram/clinical_nomogram.R
Rscript nomogram/clinical_lesion_nomogram.R
Rscript nomogram/clinical_sln_nomogram.R
Rscript nomogram/lesion_sln_nomogram.R
Rscript nomogram/comprehensive_nomogram.R
```

The R scripts write outputs under `results/nomogram/`; these files are not included in the public repository.

## Data availability and privacy

The ultrasound images, ROI annotations, patient-level clinical data, and individual prediction results used in this study are **not publicly available in this repository** due to patient privacy, ethical considerations, and applicable data-protection requirements. This release contains source code and input-format documentation only. No data-access pathway is implied or promised by this repository.

## Citation and license

If this repository accompanies a publication, add the final article citation and DOI here when available. No open-source license is asserted unless a `LICENSE` file is explicitly added by the repository owner.
