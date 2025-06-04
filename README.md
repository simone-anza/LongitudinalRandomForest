# LongitudinalRandomForest

**LongitudinalRandomForest** is a novel framework for analyzing compositional data in longitudinal studies using Random Forest models. It intelligently handles subject-level sampling to ensure one sample per subject per iteration, maximizing data independence while enabling full dataset utilization across runs.

---

## Overview

This framework is specifically designed to address key challenges in analyzing longitudinal **compositional** data—such as microbiome profiles—by:

- Ensuring **subject independence** through randomized sampling
- Handling **repeated measures** in a statistically valid way
- Utilizing **Random Forest models** for both regression and classification
- Implementing **cross-validation** with robust sampling strategies

---

## Key Features

- **Subject-Independent Sampling**  
  Each run includes only one sample per subject to avoid intra-subject correlation.

- **Full Sample Coverage**  
  All samples appear at least once across multiple randomized datasets.

- **Compositional Data Support**  
  Ideal for datasets like microbiome profiles that require special considerations.

- **Flexible ML Tasks**  
  Supports both **classification** and **regression** modes.

- **Cross-Validation Ready**  
  Implements multiple folds and repetitions for robust evaluation.

---

## Methodology

### Sample Randomization

For subjects with multiple samples (e.g., 5 timepoints per individual):

- The orchestrator creates **N random subsets** (where N ≥ number of timepoints)
- Each subset contains **one sample per subject**
- All subsets together ensure **complete sample coverage**

This design allows robust analysis while maintaining independence between samples.

---

## Analysis Workflow

1. **Split dataset** based on a variable of interest (e.g., maternal cytokine levels):
    - `Category 0`: Mid-range values (e.g., 25–75th percentile)
    - `Category 1`: Extremes (e.g., lowest 25% and highest 25%)

2. **Train models** using microbiome (or other compositional) data to predict a target (e.g., infant age).

3. **Run Random Forest** on 5 balanced subsets (1 sample/subject), repeat across both categories.

4. **Cross-Validation** perfomed with 10 folds 10 times per each subset of data.

5. **Predict external group** using models trained on the other subset.

6. **Average predictions** across folds for final output.

---

## 📂 Repository Structure

```bash
├── orchestrator.sh           # Main orchestration script
├── codes/
│   ├── regr_for_external_dataset_pred.py
│   ├── parse_ML_results.py
│   ├── create_balanced_folds.py
│   └── average_external_predictions.py
├── data/
│   ├── microbiome_data.tsv
│   └── metadata.tsv
└── results/
    └── ...
