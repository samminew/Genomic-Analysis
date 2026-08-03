import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
X = np.random.randn(50, 100)
# create a small signal for first 5 genes
X[:25, :5] += 1.5
X[25:, :5] -= 1.5

y = np.array([1]*25 + [0]*25)

df = pd.DataFrame(X, columns=[f'G{i}' for i in range(100)])
df.insert(0, 'label', y)

out = Path('tests') / 'synthetic.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out)
print(f'Wrote {out.resolve()} (shape={df.shape})')
