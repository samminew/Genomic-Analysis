"""
Comprehensive Feature Selection Pipeline
=========================================
Implements Filter, Wrapper, and Embedded methods with forward/backward selection.
Designed to work within cross-validation folds to prevent data leakage.

STANDALONE USAGE
----------------
    python feature_selection.py --dataset <path_or_name> [OPTIONS]

Arguments
---------
--dataset   Path to a CSV/TSV file  OR  a dataset name (e.g. GSE19804).
            When a name is given, the script looks for a CSV at:
              <project>/preprocessed_datasets/<name>/<name>.csv
            That file is produced by preprocessing.py's export_csv().
--label-col Name of the target/label column in the CSV (default: "label").
--n-features  Number of features to select per method (default: 20).
--methods   Space-separated subset of methods to run (default: all).
            Choices: filter_ttest  filter_anova  wrapper_svm  wrapper_rf
                     embedded_lasso
--results-dir  Directory to write results into.
            Default: <project_root>/results/feature_selection
--no-feature-files  Skip writing per-method .txt feature lists.

Examples
--------
    # Run all methods on a preprocessed dataset
    python feature_selection.py --dataset GSE19804

    # Run only fast filter methods, 50 features
    python feature_selection.py --dataset GSE19804 \\
        --n-features 50 --methods filter_ttest filter_anova embedded_lasso

    # Explicit CSV path with a custom label column
    python feature_selection.py --dataset my_data.csv --label-col diagnosis

Fix log (vs previous version)
------------------------------
- Removed module-level basicConfig call; it is now inside main() only,
  so importing this module never reconfigures the root logger.
- Removed unused `import os`.
- np.ndarray | None replaced with Optional[np.ndarray] for Python 3.9
  compatibility (covered by `from __future__ import annotations`).
- _locate_dataset now also searches the per-dataset sub-folder that
  preprocessing.export_csv() creates, closing the path-mismatch gap.
- FeatureSelectionPipeline.ALL_METHODS is the single canonical list of
  method names; svm_classifier imports it to stay in sync automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Unified feature selection interface.

    Parameters
    ----------
    method : str
        One of: filter_ttest | filter_anova | wrapper_svm | wrapper_rf |
                wrapper_forward | wrapper_backward | embedded_lasso
    n_features : int
        Number of features to select.
    """

    DISPLAY_NAMES: dict[str, str] = {
        "filter_ttest":     "Welch's T-Test (Filter)",
        "filter_anova":     "ANOVA F-Test (Filter)",
        "wrapper_svm":      "SVM-RFE (Wrapper)",
        "wrapper_rf":       "RandomForest Importance (Wrapper)",
        "embedded_lasso":   "LASSO / L1 Logistic (Embedded)",
    }

    def __init__(
        self,
        method: str = "filter_ttest",
        n_features: int = 20,
        p_value: Optional[float] = None,
        lasso_Cs: Optional[list[float]] = None,
        lasso_scoring: str = "roc_auc",
    ) -> None:
        self.method = method
        self.n_features = n_features
        # If p_value is provided, filter-based methods will select features
        # whose p-values are <= p_value instead of selecting a fixed top-k.
        # Wrapper/embedded methods still use n_features as a target.
        self.p_value = p_value
        # LASSO cross-validation grid and scoring metric
        self.lasso_Cs = np.array(lasso_Cs) if lasso_Cs is not None else None
        self.lasso_scoring = lasso_scoring
        self.selected_features: Optional[np.ndarray] = None
        self.feature_scores: Optional[np.ndarray] = None
        self.selection_rule: str = "top_k"
        # Diagnostic results for embedded LASSO CV (filled after fit)
        self.lasso_cv_results: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        """
        Fit the selector on training data.
        Must be called on the training fold ONLY to prevent data leakage.
        """
        dispatch = {
            "filter_ttest":     self._fit_ttest,
            "filter_anova":     self._fit_anova,
            "wrapper_svm":      self._fit_wrapper_svm,
            "wrapper_rf":       self._fit_wrapper_rf,
            "embedded_lasso":   self._fit_embedded_lasso,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Valid choices: {sorted(dispatch)}"
            )
        dispatch[self.method](X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return only the selected feature columns."""
        if self.selected_features is None:
            raise RuntimeError("Call fit() before transform().")
        return X[:, self.selected_features]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAMES.get(self.method, self.method)

    # ------------------------------------------------------------------
    # Filter methods
    # ------------------------------------------------------------------
    def _fit_ttest(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[filter_ttest]  Welch's Independent Two-Sample T-Test …")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("filter_ttest requires exactly 2 classes.")
        X0, X1 = X[y == classes[0]], X[y == classes[1]]
        _, p_values = ttest_ind(X1, X0, axis=0, equal_var=False)
        # If a p-value threshold is provided, select top-k features meeting p <= p_value.
        if self.p_value is not None:
            mask = np.where(p_values <= float(self.p_value))[0]
            if mask.size == 0:
                logger.warning(
                    "[filter_ttest] No features met p-value <= %s; falling back to top-%d selection",
                    self.p_value,
                    self.n_features,
                )
                self.selection_rule = f"top_k({self.n_features})"
                order = np.argsort(p_values)
                self.selected_features = order[: self.n_features]
            else:
                # Rank features satisfying p_value <= threshold by p-value and take top_k
                order = mask[np.argsort(p_values[mask])]
                self.selected_features = order[: self.n_features]
                self.selection_rule = f"top_k({len(self.selected_features)})_p<={self.p_value}"
        else:
            self.selection_rule = f"top_k({self.n_features})"
            order = np.argsort(p_values)
            self.selected_features = order[: self.n_features]
        self.feature_scores = p_values
        logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")

    def _fit_anova(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[filter_anova]  ANOVA F-Test …")
        # Compute F-statistic and associated p-values
        F, p_values = f_classif(X, y)
        # If a p-value threshold is provided, select top-k features meeting p <= p_value.
        if self.p_value is not None:
            mask = np.where(p_values <= float(self.p_value))[0]
            if mask.size == 0:
                logger.warning(
                    "[filter_anova] No features met p-value <= %s; falling back to top-%d selection",
                    self.p_value,
                    self.n_features,
                )
                self.selection_rule = f"top_k({self.n_features})"
                k = min(self.n_features, X.shape[1])
                sel = SelectKBest(score_func=f_classif, k=k)
                sel.fit(X, y)
                self.selected_features = sel.get_support(indices=True)
            else:
                # Rank features satisfying p_value <= threshold by F score descending and take top_k
                order = mask[np.argsort(-F[mask])]
                self.selected_features = order[: self.n_features]
                self.selection_rule = f"top_k({len(self.selected_features)})_p<={self.p_value}"
        else:
            self.selection_rule = f"top_k({self.n_features})"
            k = min(self.n_features, X.shape[1])
            sel = SelectKBest(score_func=f_classif, k=k)
            sel.fit(X, y)
            self.selected_features = sel.get_support(indices=True)
        self.feature_scores = F
        logger.info(f"  → {len(self.selected_features)} features selected ({self.selection_rule})")

    # ------------------------------------------------------------------
    # Wrapper helpers
    # ------------------------------------------------------------------
    def _prefilter(
        self, X: np.ndarray, y: np.ndarray, max_k: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quick ANOVA pre-filter; returns (X_filtered, original_indices)."""
        # If a p-value threshold is specified on the selector instance, prefer
        # selecting by p-value (up to max_k). Otherwise fall back to top-k ANOVA.
        if getattr(self, "p_value", None) is not None:
            _, p_values = f_classif(X, y)
            mask = np.where(p_values <= float(self.p_value))[0]
            if mask.size == 0:
                logger.warning(
                    "[prefilter] No features met p-value <= %s; falling back to top-%d prefilter",
                    self.p_value,
                    max_k,
                )
                k = min(max_k, X.shape[1])
                sel = SelectKBest(score_func=f_classif, k=k)
                X_f = sel.fit_transform(X, y)
                return X_f, sel.get_support(indices=True)
            # Cap to max_k most significant by p-value
            if mask.size > max_k:
                ordered = np.argsort(p_values)
                keep = ordered[:max_k]
                X_f = X[:, keep]
                return X_f, keep
            else:
                X_f = X[:, mask]
                return X_f, mask
        else:
            k = min(max_k, X.shape[1])
            sel = SelectKBest(score_func=f_classif, k=k)
            X_f = sel.fit_transform(X, y)
            return X_f, sel.get_support(indices=True)

    # ------------------------------------------------------------------
    # Wrapper methods
    # ------------------------------------------------------------------
    def _fit_wrapper_svm(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_svm]  SVM-RFE (with ANOVA pre-filter) …")
        X_pre, pre_idx = self._prefilter(X, y)
        logger.info(f"  Pre-filtered to {X_pre.shape[1]} features, running RFE …")
        svc = LinearSVC(
            C=0.01, penalty="l1", dual=False,
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        rfe = RFE(
            estimator=svc,
            n_features_to_select=min(self.n_features, X_pre.shape[1]),
            step=10,
        )
        rfe.fit(X_pre, y)
        self.selected_features = pre_idx[rfe.support_]
        logger.info(f"  → {len(self.selected_features)} features selected")

    def _fit_wrapper_rf(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[wrapper_rf]  RandomForest importance prefilter (with ANOVA pre-filter) …")
        X_pre, pre_idx = self._prefilter(X, y)
        logger.info(f"  Pre-filtered to {X_pre.shape[1]} features, fitting RandomForest …")
        rf = RandomForestClassifier(
            n_estimators=80, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        try:
            rf.fit(X_pre, y)
            importances = rf.feature_importances_
            # Select top-n by importance (cap to available features)
            k = min(self.n_features, X_pre.shape[1])
            order = np.argsort(importances)[-k:]
            selected_orig_idx = pre_idx[order]
            # keep them sorted for downstream reproducibility
            self.selected_features = np.sort(selected_orig_idx)
            self.feature_scores = importances
            self.selection_rule = f"wrapper_rf(top_k={k})"
            logger.info(f"  → {len(self.selected_features)} features selected (top-{k} by RF importance)")
        except Exception as exc:
            logger.warning(f"  RandomForest fit failed ({exc}); falling back to ANOVA top-k prefilter")
            # Fall back to SelectKBest on the prefiltered matrix (ANOVA F-test)
            k = min(self.n_features, X_pre.shape[1])
            sel = SelectKBest(score_func=f_classif, k=k)
            sel.fit(X_pre, y)
            keep = sel.get_support(indices=True)
            self.selected_features = np.sort(pre_idx[keep])
            # store F-scores as feature_scores when RF importances unavailable
            F, _ = f_classif(X_pre, y)
            self.feature_scores = F
            self.selection_rule = f"fallback_anova(top_k={k})"
            logger.info(f"  → {len(self.selected_features)} features selected (top-{k} by ANOVA fallback)")

    # Note: greedy forward/backward wrapper methods removed — they were
    # computationally prohibitive on high-dimensional genomic data and
    # have been replaced by pre-filtered RFE and RF importance strategies.

    # ------------------------------------------------------------------
    # Embedded methods
    # ------------------------------------------------------------------
    def _fit_embedded_lasso(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("[embedded_lasso]  LASSO (L1 Logistic Regression) …")
        # Use cross-validated LogisticRegressionCV to pick regularization strength
        # Cs is a grid of inverse regularization strengths; smaller C -> stronger regularization
        Cs = self.lasso_Cs if self.lasso_Cs is not None else np.logspace(-4, 2, 20)
        try:
            lrcv = LogisticRegressionCV(
                Cs=Cs,
                penalty="l1",
                solver="saga",
                scoring=self.lasso_scoring,
                cv=5,
                class_weight="balanced",
                random_state=42,
                max_iter=2000,
                n_jobs=-1,
                refit=True,
            )
            lrcv.fit(X, y)
            coefs = np.abs(lrcv.coef_[0])
            # Report chosen C (inverse reg strength) — note that lambda ~ 1/C
            try:
                chosen_C = lrcv.C_[0]
            except Exception:
                chosen_C = getattr(lrcv, "C_", None)
            logger.info(f"  LogisticRegressionCV selected C={chosen_C}")
            # Compute diagnostic: number of non-zero coefficients for each C
            non_zero_counts = []
            for c in Cs:
                try:
                    lr = LogisticRegression(
                        penalty="l1", solver="saga", C=float(c),
                        class_weight="balanced", max_iter=2000, random_state=42
                    )
                    lr.fit(X, y)
                    non_zero_counts.append(int((np.abs(lr.coef_[0]) > 1e-8).sum()))
                except Exception:
                    non_zero_counts.append(None)
            self.lasso_cv_results = {
                "Cs": Cs.tolist(),
                "non_zero_counts": non_zero_counts,
                "chosen_C": float(chosen_C) if chosen_C is not None else None,
            }
        except Exception as exc:
            logger.warning(f"  LogisticRegressionCV failed ({exc}); falling back to fixed-C LASSO")
            lasso = LogisticRegression(
                penalty="l1", solver="liblinear",
                C=0.1, random_state=42, max_iter=1000, class_weight="balanced",
            )
            lasso.fit(X, y)
            coefs = np.abs(lasso.coef_[0])

        non_zero = np.where(coefs > 1e-8)[0]
        if len(non_zero) == 0:
            logger.warning(
                f"  LASSO produced no non-zero coefficients; falling back to top-{self.n_features} by coefficient magnitude"
            )
            top = np.argsort(coefs)[-self.n_features:]
            self.selected_features = np.sort(top)
        elif len(non_zero) <= self.n_features:
            logger.info(
                f"  LASSO found {len(non_zero)} non-zero coefficients (target was {self.n_features}); keeping all non-zero."
            )
            self.selected_features = non_zero
        else:
            top = np.argsort(coefs)[-self.n_features:]
            self.selected_features = np.sort(top)

        self.feature_scores = coefs
        logger.info(f"  → {len(self.selected_features)} features selected")


# ---------------------------------------------------------------------------
# FeatureSelectionPipeline
# ---------------------------------------------------------------------------
class FeatureSelectionPipeline:
    """
    Orchestrates all feature selection methods on the same dataset.

    ALL_METHODS is the single canonical list of valid method names.
    svm_classifier imports it directly so the two files never drift apart.
    """

    # Single source of truth for valid method names — imported by svm_classifier
    ALL_METHODS: list[str] = list(FeatureSelector.DISPLAY_NAMES.keys())

    def __init__(
        self,
        n_features: int = 20,
        methods: Optional[list[str]] = None,
        p_value: Optional[float] = None,
        lasso_Cs: Optional[list[float]] = None,
        lasso_scoring: str = "roc_auc",
    ) -> None:
        self.n_features = n_features
        chosen = methods or self.ALL_METHODS
        unknown = set(chosen) - set(self.ALL_METHODS)
        if unknown:
            raise ValueError(
                f"Unknown method(s): {unknown}. "
                f"Valid choices: {self.ALL_METHODS}"
            )
        self.p_value = p_value
        self.lasso_Cs = lasso_Cs
        self.lasso_scoring = lasso_scoring
        self.selectors: dict[str, FeatureSelector] = {
            m: FeatureSelector(m, n_features, p_value=p_value, lasso_Cs=lasso_Cs, lasso_scoring=lasso_scoring)
            for m in chosen
        }

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit all selectors; returns a results dict."""
        logger.info("=" * 65)
        logger.info("FEATURE SELECTION PIPELINE — fitting all methods")
        logger.info("=" * 65)
        results: dict = {}
        for method, selector in self.selectors.items():
            try:
                selector.fit(X, y)
                results[method] = {
                    "n_selected": int(len(selector.selected_features)),
                    "selected_indices": selector.selected_features,
                }
            except Exception as exc:
                logger.error(f"  ERROR in {method}: {exc}")
                results[method] = {"error": str(exc)}
        return results


# ---------------------------------------------------------------------------
# Data loading (standalone mode)
# ---------------------------------------------------------------------------
def _locate_dataset(dataset_arg: str, script_dir: Path) -> Path:
    """
    Resolve --dataset to an actual CSV file path.

    Accepted forms
    --------------
    1. An absolute or relative path to an existing .csv / .tsv file.
    2. A dataset name — looks in these locations in order:
         <project>/preprocessed_datasets/<name>/<name>.csv   ← export_csv() output
         <project>/preprocessed_datasets/<name>.csv
         <project>/datasets/<name>/<name>.csv
         <project>/datasets/<name>.csv
    """
    p = Path(dataset_arg)
    if p.exists():
        return p.resolve()

    project_dir = script_dir.parent
    candidates = [
        project_dir / "preprocessed_datasets" / p.name / f"{p.name}.csv",
        project_dir / "preprocessed_datasets" / f"{p.name}.csv",
        project_dir / "datasets" / p.name / f"{p.name}.csv",
        project_dir / "datasets" / f"{p.name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Resolved '{dataset_arg}' → {candidate}")
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find dataset '{dataset_arg}'.\n"
        "Tried:\n" + "\n".join(f"  {c}" for c in candidates) + "\n"
        "Run preprocessing.py first to generate the CSV, or pass a direct path."
    )


def load_dataset(
    csv_path: Path,
    label_col: str = "label",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load a CSV/TSV dataset; split into X (features) and y (labels).

    The label column must contain binary values (0/1) or exactly two
    unique values that will be label-encoded to 0/1.
    """
    sep = "\t" if csv_path.suffix.lower() in {".tsv", ".txt"} else ","
    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, sep=sep, index_col=0)
    logger.info(f"  Shape: {df.shape}")

    # Case-insensitive column lookup
    if label_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == label_col.lower()]
        if not matches:
            raise KeyError(
                f"Label column '{label_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        label_col = matches[0]
        logger.info(f"  Using label column: '{label_col}'")

    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    unique_vals = sorted(y_raw.unique())
    if len(unique_vals) != 2:
        raise ValueError(
            f"Expected exactly 2 class labels, found {len(unique_vals)}: {unique_vals}"
        )
    if set(unique_vals) != {0, 1}:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        logger.info(f"  Encoding labels: {mapping}")
        y_raw = y_raw.map(mapping)

    y = y_raw.to_numpy(dtype=int)
    logger.info(
        f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}  |  "
        f"Class distribution: {np.bincount(y).tolist()}"
    )
    return X, y


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _print_console_summary(
    results: dict,
    n_original: int,
    selectors: dict[str, FeatureSelector],
) -> None:
    w = 72
    print("\n" + "=" * w)
    print("FEATURE SELECTION — RESULTS SUMMARY")
    print("=" * w)
    print(f"Original feature space: {n_original} features\n")
    print(f"{'Method':<22} {'Display Name':<32} {'Selected':>8} {'Reduction':>10}")
    print("-" * w)
    for method, res in results.items():
        display = selectors[method].display_name
        if "error" in res:
            print(f"{method:<22} {display:<32} {'ERROR':>8}  {res['error']}")
        else:
            n_sel = res["n_selected"]
            pct = (1 - n_sel / n_original) * 100
            print(f"{method:<22} {display:<32} {n_sel:>8} {pct:>9.1f}%")
    print("=" * w + "\n")


def save_results(
    pipeline: FeatureSelectionPipeline,
    results: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    dataset_name: str,
    results_base_dir: Path,
    n_features: int,
    write_feature_files: bool = True,
) -> Path:
    """
    Write all output files into results_base_dir / dataset_name /

    Files created
    -------------
    feature_selection_summary_<dataset>_<ts>.csv
        One row per method: counts, reduction %, feature indices and names.
    feature_selection_results_<dataset>_<ts>.json
        Full structured results with metadata block.
    feature_selection_report_<dataset>_<ts>.txt
        Human-readable text report.
    selected_features_<dataset>_<ts>/
        <method>_selected_features.txt   — one file per method.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_base_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # 1. CSV summary
    # ------------------------------------------------------------------
    csv_rows = []
    for method, selector in pipeline.selectors.items():
        if selector.selected_features is not None:
            idx = sorted(selector.selected_features.tolist())
            names = [feature_names[i] for i in idx]
            csv_rows.append({
                "Method": method,
                "Display Name": selector.display_name,
                "Features Selected": len(idx),
                "Original Features": X.shape[1],
                "Reduction (%)": round((1 - len(idx) / X.shape[1]) * 100, 2),
                "Selected Feature Indices": ";".join(map(str, idx)),
                "Selected Feature Names": ";".join(names),
            })
        else:
            csv_rows.append({
                "Method": method,
                "Display Name": selector.display_name,
                "Features Selected": "ERROR",
                "Original Features": X.shape[1],
                "Reduction (%)": "",
                "Selected Feature Indices": results[method].get("error", ""),
                "Selected Feature Names": "",
            })

    csv_path = out_dir / f"feature_selection_summary_{dataset_name}_{ts}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  ✓  CSV summary          →  {csv_path.name}")

    # ------------------------------------------------------------------
    # 2. JSON results
    # ------------------------------------------------------------------
    json_data: dict = {
        "metadata": {
            "dataset": dataset_name,
            "timestamp": ts,
            "n_samples": int(X.shape[0]),
            "n_original_features": int(X.shape[1]),
            "n_target_features": n_features,
            "class_distribution": np.bincount(y).tolist(),
        },
        "methods": {},
    }
    for method, selector in pipeline.selectors.items():
        if selector.selected_features is not None:
            idx = sorted(selector.selected_features.tolist())
            json_data["methods"][method] = {
                "display_name": selector.display_name,
                "n_selected": len(idx),
                "reduction_percent": round(
                    (1 - len(idx) / X.shape[1]) * 100, 2
                ),
                "selected_indices": idx,
                "selected_feature_names": [feature_names[i] for i in idx],
            }
        else:
            json_data["methods"][method] = {
                "display_name": selector.display_name,
                "error": results[method].get("error", "unknown error"),
            }

    json_path = out_dir / f"feature_selection_results_{dataset_name}_{ts}.json"
    with open(json_path, "w") as fh:
        json.dump(json_data, fh, indent=2)
    print(f"  ✓  JSON results         →  {json_path.name}")

    # ------------------------------------------------------------------
    # 3. Text report
    # ------------------------------------------------------------------
    report_path = out_dir / f"feature_selection_report_{dataset_name}_{ts}.txt"
    with open(report_path, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write("FEATURE SELECTION EVALUATION REPORT\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(f"Dataset      : {dataset_name}\n")
        fh.write(f"Timestamp    : {ts}\n")
        fh.write(f"Samples      : {X.shape[0]}\n")
        fh.write(f"Features     : {X.shape[1]}\n")
        fh.write(f"Target n_feat: {n_features}\n")
        dist = np.bincount(y)
        fh.write(f"Class dist.  : {dist[0]} class-0 / {dist[1]} class-1\n\n")
        fh.write("=" * 70 + "\n")
        fh.write("RESULTS BY METHOD\n")
        fh.write("=" * 70 + "\n\n")
        for method, selector in pipeline.selectors.items():
            fh.write(f"  Method  : {selector.display_name}\n")
            fh.write(f"  Key     : {method}\n")
            if selector.selected_features is not None:
                idx = sorted(selector.selected_features.tolist())
                names = [feature_names[i] for i in idx]
                pct = (1 - len(idx) / X.shape[1]) * 100
                fh.write(
                    f"  Selected: {len(idx)} / {X.shape[1]} features"
                    f"  ({pct:.1f}% reduction)\n"
                )
                fh.write(f"  Indices : {idx}\n")
                fh.write(f"  Names   : {names}\n")
            else:
                fh.write(f"  ERROR   : {results[method].get('error', '?')}\n")
            fh.write("\n" + "-" * 70 + "\n\n")
    print(f"  ✓  Text report          →  {report_path.name}")

    # ------------------------------------------------------------------
    # 4. Per-method feature list files
    # ------------------------------------------------------------------
    if write_feature_files:
        feat_dir = out_dir / f"selected_features_{dataset_name}_{ts}"
        feat_dir.mkdir(parents=True, exist_ok=True)
        for method, selector in pipeline.selectors.items():
            fname = feat_dir / f"{method}_selected_features.txt"
            with open(fname, "w") as fh:
                fh.write(f"Feature Selection Method : {selector.display_name}\n")
                fh.write(f"Method Key               : {method}\n")
                fh.write(f"Dataset                  : {dataset_name}\n")
                fh.write(f"Timestamp                : {ts}\n")
                fh.write(f"Original Feature Space   : {X.shape[1]}\n")
                if selector.selected_features is not None:
                    idx = sorted(selector.selected_features.tolist())
                    names = [feature_names[i] for i in idx]
                    pct = (1 - len(idx) / X.shape[1]) * 100
                    fh.write(f"Features Selected        : {len(idx)}\n")
                    fh.write(f"Dimensionality Reduction : {pct:.2f}%\n")
                    fh.write("=" * 60 + "\n")
                    fh.write(f"{'Index':<10} Feature Name\n")
                    fh.write("-" * 60 + "\n")
                    for i, name in zip(idx, names):
                        fh.write(f"{i:<10} {name}\n")
                else:
                    fh.write(f"ERROR: {results[method].get('error', 'unknown')}\n")
        print(f"  ✓  Per-method .txt files →  {feat_dir.name}/")

    # ------------------------------------------------------------------
    # 5. LASSO diagnostic plot (if available)
    # ------------------------------------------------------------------
    for method, selector in pipeline.selectors.items():
        if method == "embedded_lasso" and getattr(selector, "lasso_cv_results", None) is not None:
            diagnostics = selector.lasso_cv_results
            try:
                Cs = diagnostics["Cs"]
                counts = diagnostics["non_zero_counts"]
                plt.figure()
                plt.plot(Cs, counts, marker="o")
                plt.xscale("log")
                plt.xlabel("C (inverse regularization)")
                plt.ylabel("Number of non-zero coefficients")
                plt.title(f"LASSO: Non-zero coefficients vs C ({dataset_name})")
                plt.grid(True)
                plot_path = out_dir / f"lasso_nonzero_vs_C_{dataset_name}_{ts}.png"
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
                print(f"  ✓  LASSO diagnostic plot →  {plot_path.name}")
            except Exception as exc:
                print(f"  !  Could not create LASSO diagnostic plot: {exc}")

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help=(
            "Path to a CSV/TSV file OR a dataset name whose preprocessed CSV "
            "lives under <project>/preprocessed_datasets/<name>/<name>.csv"
        ),
    )
    parser.add_argument(
        "--label-col", "-l",
        default="label",
        dest="label_col",
        help="Name of the target/label column in the CSV (default: 'label')",
    )
    parser.add_argument(
        "--n-features", "-n",
        type=int,
        default=20,
        dest="n_features",
        help="Number of features to select per method (default: 20)",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=None,
        dest="p_value",
        help=(
            "P-value threshold for filter methods. If provided, filter methods "
            "(t-test, ANOVA) will select all genes with p <= p-value instead of "
            "selecting a fixed top-k. Default: None (use top-k selection)."
        ),
    )
    parser.add_argument(
        "--lasso-cs",
        type=str,
        default=None,
        dest="lasso_cs",
        help=(
            "Comma-separated list of C values for LASSO CV (e.g. '1e-4,1e-3,1e-2'). "
            "If omitted, a logspace grid is used."
        ),
    )
    parser.add_argument(
        "--lasso-scoring",
        type=str,
        default="roc_auc",
        dest="lasso_scoring",
        help="Scoring metric for LASSO CV (default: 'roc_auc')",
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        choices=FeatureSelectionPipeline.ALL_METHODS,
        metavar="METHOD",
        help=(
            "Methods to run (default: all). "
            f"Choices: {', '.join(FeatureSelectionPipeline.ALL_METHODS)}"
        ),
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=None,
        dest="results_dir",
        help=(
            "Directory to write results into. "
            "Defaults to <project_root>/results/feature_selection"
        ),
    )
    parser.add_argument(
        "--no-feature-files",
        action="store_true",
        dest="no_feature_files",
        help="Skip writing individual per-method feature list .txt files",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    # Configure logging here — not at module level — so importing this file
    # never reconfigures the root logger when used as a library.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent  # …/scripts/

    try:
        csv_path = _locate_dataset(args.dataset, script_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    dataset_name = csv_path.stem

    try:
        X, y = load_dataset(csv_path, label_col=args.label_col)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        logger.error(f"Failed to load dataset: {exc}")
        return 1

    results_base_dir = (
        Path(args.results_dir)
        if args.results_dir
        else script_dir.parent / "results" / "feature_selection"
    )

    print("\n" + "=" * 65)
    print("FEATURE SELECTION EVALUATION TOOL")
    print("=" * 65)
    print(f"  Dataset    : {dataset_name}  ({csv_path})")
    print(f"  Shape      : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Label col  : '{args.label_col}'")
    print(f"  n_features : {args.n_features}")
    print(f"  Methods    : {args.methods or 'all'}")
    print(f"  Output dir : {results_base_dir / dataset_name}")
    print("=" * 65 + "\n")

    # Parse lasso Cs if provided
    lasso_Cs = None
    if args.lasso_cs:
        try:
            lasso_Cs = [float(x) for x in args.lasso_cs.split(",") if x.strip()]
        except Exception:
            logger.warning("Could not parse --lasso-cs; falling back to default grid")

    pipeline = FeatureSelectionPipeline(
        n_features=args.n_features,
        methods=args.methods,
        p_value=args.p_value,
        lasso_Cs=lasso_Cs,
        lasso_scoring=args.lasso_scoring,
    )
    results = pipeline.fit_all(X.to_numpy(), y)

    _print_console_summary(results, X.shape[1], pipeline.selectors)

    print("Saving results …")
    out_dir = save_results(
        pipeline=pipeline,
        results=results,
        X=X,
        y=y,
        dataset_name=dataset_name,
        results_base_dir=results_base_dir,
        n_features=args.n_features,
        write_feature_files=not args.no_feature_files,
    )

    print(f"\nAll results saved to: {out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
