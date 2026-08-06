import sys
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Ensure scripts are importable
scripts_dir = str(Path(__file__).resolve().parent.parent / 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from feature_selection import FeatureSelector


def test_path_b_filter_ttest_one_fold():
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label'])

    # Use a single train/test split (first half / second half) for speed
    n = len(df)
    split = n // 2
    X_train = X.iloc[:split]
    y_train = y[:split]
    X_test = X.iloc[split:]
    y_test = y[split:]

    pipeline = Pipeline([
        ('selector', FeatureSelector(method='filter_ttest', n_features=10)),
        ('svm', SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42))
    ])

    # Should not raise
    pipeline.fit(X_train.values, y_train)
    y_pred = pipeline.predict(X_test.values)

    assert hasattr(pipeline.named_steps['selector'], 'selected_features')
    sel = pipeline.named_steps['selector'].selected_features
    assert sel is not None and len(sel) > 0
    assert len(y_pred) == len(y_test)


def test_selector_reports_pvalue_selection_rule():
    selector = FeatureSelector(method='filter_ttest', n_features=5, p_value=0.05)
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label']).to_numpy()
    selector.fit(X, y)
    assert 'p<=0.05' in selector.selection_rule


def test_wrapper_rf_fallback_returns_features():
    df = pd.read_csv('tests/synthetic.csv', index_col=0)
    y = df['label'].to_numpy(dtype=int)
    X = df.drop(columns=['label']).to_numpy()

    selector = FeatureSelector(method='wrapper_rf', n_features=5, p_value=0.05)
    selector.fit(X, y)

    assert selector.selected_features is not None
    assert len(selector.selected_features) > 0
