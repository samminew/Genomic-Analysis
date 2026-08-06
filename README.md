# Effect of Feature Selection on Classification Accuracy in High-Dimensional Gene Expression Data

## Project Overview

This project investigates the comparative impact of different feature selection techniques on machine learning classification accuracy in high-dimensional gene expression datasets. The research addresses the **small *n*, large *p*** problem prevalent in genomics, where datasets contain tens of thousands of gene probes but relatively few samples.

**Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Kumasi  
**Department:** Mathematics  
**Degree:** BSc. Mathematics  
**Authors:** Darko Samuel and Akyen Samuel

## Research Objectives

1. **Analyze** high-dimensional gene expression data structure and characteristics
2. **Implement** three feature selection approaches:
   - Filter methods (Welch's T-Test, ANOVA F-Test)
   - Wrapper methods (SVM-RFE, Random Forest importance)
   - Embedded methods (LASSO / L1-regularized logistic regression)
3. **Train** an SVM classifier on selected features using leakage-free cross-validation
4. **Evaluate** classification performance across all feature subsets
5. **Determine** which feature selection technique most effectively improves classification accuracy

## Project Status

### Completed

- **Preprocessing pipeline** (`scripts/preprocessing.py`): GEO download, log2 transformation, z-score normalization, `.npy` cache + CSV export
- **Feature selection pipeline** (`scripts/feature_selection.py`): all five methods with dual selection rules (top-*k* or p-value threshold)
- **Feature selection runs** on GSE19804 and GSE42568 (results under `results/feature_selection/`)
- **SVM evaluation framework** (`scripts/svm_classifier.py`, `scripts/svm_comparative.py`): Path A baseline + Path B with feature selection inside each CV fold
- **Integration tests** (`tests/test_svm_integration.py`)
- **Thesis draft**: Chapters 1–4 in LaTeX (`chapters/`)

### In Progress

- **SVM classifier evaluation**: run full pipeline on both datasets and populate Chapter 4 with quantitative results
- **Experimental consistency**: align selection rules (top-*k* vs p-value threshold) across datasets before final comparison

### Pending

- Chapter 5: Conclusions and Recommendations
- Final report PDF compilation
- Optional: Random Forest and Logistic Regression classifiers (out of current scope; thesis focuses on SVM)

## Datasets

Both datasets are registered in `GenomicDataProcessor.DATASET_CONFIG` and share the same high-dimensional microarray structure (~54,675 probes).

| Dataset | Cancer type | Samples | Class balance | p/n ratio |
|---------|-------------|---------|---------------|-----------|
| **GSE19804** | Lung | 120 (60 / 60) | Balanced | ~455:1 |
| **GSE42568** | Breast | 121 (104 / 17) | Imbalanced | ~452:1 |

**Source:** [NCBI Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/)

GSE19804 is the primary feature-selection dataset; GSE42568 provides a secondary benchmark under severe class imbalance.

## Pipeline Overview

The canonical workflow is script-based. The Jupyter notebook (`Genomic_data_analysis (3).ipynb`) is retained as early exploratory work but is no longer the primary pipeline.

```
preprocessing.py  →  feature_selection.py  →  svm_classifier.py
        │                      │                      │
        ▼                      ▼                      ▼
preprocessed_datasets/   results/feature_selection/   results/<dataset_folder>/
```

### Step 1 — Preprocessing

Downloads (or loads cached) GEO series matrices, applies log2(*x* + 1) and per-feature z-score normalization, and writes:

- NumPy cache: `preprocessed_datasets/<dataset>/`
- Labelled CSV: `preprocessed_datasets/<dataset>/<dataset>.csv`

### Step 2 — Feature Selection

Runs filter, wrapper, and embedded selectors. Wrapper methods use an ANOVA prefilter (top 500 by default) before SVM-RFE or Random Forest ranking.

### Step 3 — SVM Evaluation

- **Path A (baseline):** linear SVM on all features, no selection
- **Path B (optimised):** `FeatureSelector` → `SVC` inside a `sklearn.pipeline.Pipeline`, fitted only on each training fold

Evaluation uses **5-fold stratified cross-validation** with balanced class weights. Primary metric: **Matthews Correlation Coefficient (MCC)**.

## Experimental Settings

### Feature selection rules

The pipeline supports two selection modes, controlled by CLI flags:

| Mode | Flag | Behaviour |
|------|------|-----------|
| **Top-*k*** (default) | `--n-features 20` | Retain the 20 most significant features (filters) or run RFE/LASSO to 20 features (wrappers/embedded) |
| **P-value threshold** | `--p-value 0.05` | Filter methods retain **all** probes with *p* ≤ threshold; wrappers cap the prefilter at 500 |

**Methods implemented:** `filter_ttest`, `filter_anova`, `wrapper_svm`, `wrapper_rf`, `embedded_lasso`

Greedy forward/backward wrapper selection was removed for computational tractability.

### Current runs (August 2026)

| Dataset | Filter methods | Wrapper / LASSO | Notes |
|---------|----------------|-----------------|-------|
| GSE19804 | 20 features each | 20 features each | Standardized `--n-features 20 --p-value 0.05` |
| GSE42568 | 20 features each | 20 features each | Standardized `--n-features 20 --p-value 0.05` |

> **Note:** Both datasets are evaluated under identical selection parameters ($p \le 0.05, k = 20$), providing a direct cross-dataset and cross-method comparison.

### SVM settings

- Kernel: linear
- `C = 1.0`, `class_weight = 'balanced'`
- 5-fold stratified CV, `random_state = 42`
- Feature selection applied **inside each fold** (no data leakage)

## Directory Structure

```
GenomicsProj/
├── main.tex                              # LaTeX master document
├── chapters/
│   ├── chapter1_introduction.tex
│   ├── chapter2_methodology.tex
│   ├── chapter3_results.tex
│   └── chapter4_svm_evaluation.tex
├── scripts/
│   ├── preprocessing.py                  # Download, transform, cache, export CSV
│   ├── feature_selection.py              # Filter / wrapper / embedded selectors
│   ├── svm_classifier.py                 # Single-dataset SVM evaluation
│   └── svm_comparative.py                # Run SVM on all registered datasets
├── tests/
│   ├── test_svm_integration.py           # Pipeline smoke tests
│   ├── make_synthetic.py                 # Synthetic data generator
│   ├── run_svm_synthetic.py              # Quick SVM test on synthetic data
│   └── synthetic.csv
├── results/
│   ├── README.md                         # SVM results layout guide
│   └── feature_selection/                # Per-dataset feature selection outputs
│       ├── GSE19804/
│       └── GSE42568/
├── datasets/                             # Raw GEO downloads (gitignored)
├── preprocessed_datasets/                # Cached .npy + CSV (gitignored)
├── Genomic_data_analysis (3).ipynb       # Legacy exploratory notebook
├── requirements.txt
└── venv_python_runner.bat                # Windows helper for venv Python
```

## Running the Analysis

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

Or install the full pinned environment:

```bash
pip install -r requirements.txt
```

### 1. Preprocess a dataset

```bash
python scripts/preprocessing.py GSE19804
python scripts/preprocessing.py GSE42568
```

### 2. Run feature selection

Top-*k* mode (20 features per method — recommended for comparable subsets):

```bash
python scripts/feature_selection.py --dataset GSE19804 --n-features 20
python scripts/feature_selection.py --dataset GSE42568 --n-features 20
```

P-value threshold mode (filters retain all significant probes):

```bash
python scripts/feature_selection.py --dataset GSE19804 --p-value 0.05 --n-features 20
```

Run a subset of methods:

```bash
python scripts/feature_selection.py --dataset GSE19804 \
    --methods filter_ttest filter_anova embedded_lasso
```

### 3. Train and evaluate SVM

Single dataset (default: GSE42568):

```bash
python scripts/svm_classifier.py --dataset GSE19804
python scripts/svm_classifier.py --dataset GSE42568 --n-features 20 --p-value 0.05
```

All registered datasets:

```bash
python scripts/svm_comparative.py --n-splits 5
```

Run only baseline or a single Path B method:

```bash
python scripts/svm_classifier.py --dataset GSE19804 --run path_a
python scripts/svm_classifier.py --dataset GSE19804 --run path_b --method filter_ttest
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Compile the thesis

```bash
pdflatex main.tex
pdflatex main.tex   # Run twice for table of contents
```

## Outputs

### Feature selection (`results/feature_selection/<dataset>/`)

| File | Description |
|------|-------------|
| `feature_selection_summary_<timestamp>.csv` | Method, feature count, reduction %, selected gene names |
| `feature_selection_results_<timestamp>.json` | Full structured results |
| `feature_selection_report_<timestamp>.txt` | Human-readable report |
| `selected_features_<timestamp>/` | Per-method `.txt` gene lists |

### SVM evaluation (`results/<dataset>_<cancer_type>/`)

| File | Description |
|------|-------------|
| `svm_training.log` | Per-fold training progress |
| `svm_results_<timestamp>.json` | Path A + all Path B summaries |
| `svm_summary_<timestamp>.csv` | Metrics table (accuracy, MCC, F1, etc.) |

SVM result files are gitignored; generate them locally by running the classifier scripts.

## Evaluation Metrics

For each feature subset (Path A + five Path B methods), the pipeline reports:

- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Matthews Correlation Coefficient (MCC)** — primary metric, especially for GSE42568
- **ROC-AUC** (when probability estimates are available)
- **Sensitivity / Specificity** and confusion matrix counts (TP, TN, FP, FN)

All metrics are aggregated as mean ± std across CV folds.

## References

**Dataset source:** NCBI GEO — [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804), [GSE42568](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568)

**Python libraries:** scikit-learn, pandas, numpy, scipy, matplotlib, seaborn

## Notes

- The small-*n*, large-*p* setting (p/n ≈ 450:1) makes overfitting inevitable without feature selection; baseline SVM on the full feature set is expected to perform poorly.
- Preprocessing (log2 + z-score) is applied to the full dataset before CV; the scaler is not refit per fold — a known simplification documented in the thesis.
- Filter methods with a permissive p-value threshold (e.g. 0.05) can retain thousands of probes; wrapper and embedded methods always produce compact subsets.
- Choose one selection policy (top-*k* or p-value) and apply it consistently across datasets before drawing final comparative conclusions.
