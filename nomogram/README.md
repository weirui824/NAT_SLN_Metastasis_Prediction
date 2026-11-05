# Nomogram and statistical modeling (R)

These scripts implement the R-based statistical analyses used after independent Python breast lesion and SLN model training. The study team has confirmed that the scripts run successfully on the private research dataset and agree with the final analysis. Some source files retain comments identifying their reconstruction from archived analysis materials.

## Environment

- R 4.5.0
- `rms` 8.0-0
- `readr` 2.1.5
- `pROC` (required; exact version not recorded)

## Files

| Script | Analysis |
| --- | --- |
| `common.R` | Private data loading, logit conversion, `rms` logistic regression, nomogram construction, bootstrap calibration, AUC calculation, and local outputs |
| `clinical_nomogram.R` | Five clinical predictors |
| `clinical_lesion_nomogram.R` | Five clinical predictors plus lesion imaging probability logit |
| `clinical_sln_nomogram.R` | Five clinical predictors plus SLN imaging probability logit |
| `lesion_sln_nomogram.R` | Lesion and SLN probability logits |
| `comprehensive_nomogram.R` | Logistic regression fusion of the two imaging probability logits, followed by a comprehensive model with five clinical predictors |

## Private inputs

`common.R` reads `data/processed/train_all.csv` and `data/processed/test_all.csv`, which are **not provided**. Depending on the script, required columns include `patient_id`, `sln_id`, `label`, `Ki_67`, `SLN_diameter`, `HER_2`, `cortex_thickness`, `N_grade_pred`, `DL_lesion`, and `DL_SLN`. `label` is binary (0/1), and the model inputs are expected to be numeric.

Ensure that imaging probabilities are correctly matched to the same observations, and that logistic regression fitting does not use information from the independent test set. The supplied code fits the imaging fusion on the training data; the source of the base-model training predictions must be checked to assess possible stacking leakage.

## Run

From the repository root, for example:

```bash
Rscript nomogram/clinical_nomogram.R
Rscript nomogram/clinical_lesion_nomogram.R
Rscript nomogram/clinical_sln_nomogram.R
Rscript nomogram/lesion_sln_nomogram.R
Rscript nomogram/comprehensive_nomogram.R
```

Outputs are saved locally under `results/nomogram/`, including nomogram plots, AUC summaries, predictions, calibration plots, and fitted models. They are excluded from the public repository. Python, rather than R, is used for the deep learning ROC curve visualization; `pROC` remains a dependency because `common.R` computes nomogram AUC values.
