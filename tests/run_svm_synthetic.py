import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Ensure sibling scripts are importable
scripts_dir = str(Path(__file__).resolve().parent.parent / 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from feature_selection import FeatureSelector, FeatureSelectionPipeline


def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:,1])
        except Exception:
            metrics['roc_auc'] = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
    return metrics


if __name__ == '__main__':
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label'])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Path A: baseline
    fold_results = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        svm = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_proba = svm.predict_proba(X_test)
        m = evaluate(y_test, y_pred, y_proba)
        m['n_features'] = X.shape[1]
        fold_results.append(m)
    print('Path A baseline summary:')
    df_a = pd.DataFrame(fold_results)
    print(df_a.mean())

    # Path B: optimized per method
    for method in FeatureSelectionPipeline.ALL_METHODS:
        print('\nRunning Path B method:', method)
        fold_results = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipeline = Pipeline([
                ('selector', FeatureSelector(method=method, n_features=10)),
                ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42))
            ])
            try:
                pipeline.fit(X_train.values, y_train)
                y_pred = pipeline.predict(X_test.values)
                y_proba = pipeline.predict_proba(X_test.values)
                m = evaluate(y_test, y_pred, y_proba)
                m['n_features'] = len(pipeline.named_steps['selector'].selected_features)
                fold_results.append(m)
            except Exception as e:
                print('  Method failed on fold:', e)
                fold_results = None
                break
        if fold_results:
            dfb = pd.DataFrame(fold_results)
            print(dfb.mean())
        else:
            print('  Skipping method due to errors')
