#!/usr/bin/env python
"""Regenerate Table 4.1 data with all variants and correct metrics."""

from scripts.svm_classifier import SVMClassifierWithCV
import json

# Run all variants for filter_ttest only
c = SVMClassifierWithCV(dataset_name='GSE42568', n_splits=5, n_features=20, p_value=0.05)

variants = [
    ('baseline', dict(apply_smote=False, tune_threshold=False, tune_c=False, alternative_classifier=None)),
    ('smote', dict(apply_smote=True, tune_threshold=False, tune_c=False, alternative_classifier=None)),
    ('threshold_tuned', dict(apply_smote=False, tune_threshold=True, tune_c=False, alternative_classifier=None)),
    ('c_tuned', dict(apply_smote=False, tune_threshold=False, tune_c=True, alternative_classifier=None)),
    ('logreg', dict(apply_smote=False, tune_threshold=False, tune_c=False, alternative_classifier='logistic_regression')),
    ('rf', dict(apply_smote=False, tune_threshold=False, tune_c=False, alternative_classifier='random_forest')),
]

results = {}
for label, kwargs in variants:
    print(f'Running {label}...')
    c.train_path_b_optimized(feature_method='filter_ttest', n_features=20, p_value=0.05, **kwargs)
    # Get result key
    if kwargs['tune_threshold']:
        key = 'path_b_filter_ttest_threshold_tuned'
    elif kwargs['tune_c']:
        key = 'path_b_filter_ttest_c_tuned'
    elif kwargs['alternative_classifier']:
        key = f'path_b_filter_ttest_{kwargs["alternative_classifier"]}'
    else:
        key = 'path_b_filter_ttest'
    
    if key in c.results:
        r = c.results[key]['summary']
        results[label] = {
            'accuracy': f"{r.get('accuracy_mean', 0):.4f} ± {r.get('accuracy_std', 0):.4f}",
            'recall': f"{r.get('recall_mean', 0):.4f} ± {r.get('recall_std', 0):.4f}",
            'specificity': f"{r.get('specificity_mean', 0):.4f} ± {r.get('specificity_std', 0):.4f}",
            'f1': f"{r.get('f1_mean', 0):.4f} ± {r.get('f1_std', 0):.4f}",
            'mcc': f"{r.get('mcc_mean', 0):.4f} ± {r.get('mcc_std', 0):.4f}",
            'roc_auc': f"{r.get('roc_auc_mean', 0):.4f} ± {r.get('roc_auc_std', 0):.4f}",
        }
        print(f'{label}: done')
    else:
        print(f'Warning: {key} not found in results')

print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))

# Also create LaTeX table code
print("\n=== LaTeX TABLE CODE ===")
for label, metrics in results.items():
    print(f"{label.replace('_', ' ').title()} & {metrics['accuracy']} & {metrics['recall']} & {metrics['specificity']} & {metrics['f1']} & {metrics['mcc']} & {metrics['roc_auc']} \\\\")
