# Sample data

Small files for trying the app without training anything.

| File | Use it as |
|---|---|
| `sample_classifier.pkl` | A model. Logistic regression, expects columns `f1`, `f2`. |
| `sample_classification.csv` | A dataset for it. Target column: `label`. Choose the **Classification** task. |
| `sample_regressor.pkl` | A model. Linear regression, expects columns `f1`, `f2`. |
| `sample_regression.csv` | A dataset for it. Target column: `label`. Choose the **Regression** task. |
| `bad_dataset_missing_target.csv` | A dataset that fails validation, to see an error message. |

The models were saved with scikit-learn 1.x. If a model fails to load after you
upgrade scikit-learn, re-create it with the snippet in `docs/USAGE.md`.
