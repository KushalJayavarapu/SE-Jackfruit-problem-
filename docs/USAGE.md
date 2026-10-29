# User Guide

How to prepare files for the Model Evaluator, run an evaluation, read the
results and fix common problems. For installation, see the
[main README](../README.md).

## 1. Prepare your files

### The model

The model must be a trained scikit-learn style object saved as `.pkl` or
`.joblib`. It needs a `predict()` method. For ROC-AUC on classifiers it also
needs `predict_proba()`.

Example that creates a compatible classifier (needs scikit-learn and pandas,
already installed with the app):

```python
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LogisticRegression(max_iter=5000).fit(X_train, y_train)   # train on a DataFrame

joblib.dump(model, "classifier.joblib")
X_test.assign(target=y_test).to_csv("test_data.csv", index=False)
```

Tips:

- **Train on a pandas DataFrame** (as above). The model then remembers its column
  names and the app matches them to the dataset by name, so column order does
  not matter. If you train on a plain NumPy array, the app can only check that
  the column count matches.
- **Use the same scikit-learn version** to save and to load. A model saved with a
  very different version may fail to load.
- **Save only the model**, not a dictionary or tuple holding the model.
- If your model needs preprocessing (scaling, encoding), save the whole
  `Pipeline` so the raw dataset columns work as they are.

### The dataset

- A `.csv` file with a header row, up to **50,000 rows**.
- One column is the **target** (the true answer). Every other column is a feature.
- The target column must have **no empty values**.
- The feature columns must be the ones the model was trained on. Extra columns
  are allowed: they are ignored and listed in the result notes.
- For classification, target values must be the same labels the model learned.
  For regression, the target must be numeric.

## 2. Run an evaluation

1. **Upload the model.** In *Add and validate a model*, choose the file and press
   **Upload model**.
2. **Validate the model.** Pick it in *Model to validate* and press
   **Validate model**. A valid model shows its type and the inputs it expects.
3. **Upload the dataset.** In *Add and validate a dataset*, choose the CSV, type
   the **exact** target column name (case-sensitive) and press **Upload dataset**.
4. **Validate the dataset.** Pick it and press **Validate dataset**. A valid
   dataset shows its row count and a per-column summary.
5. **Evaluate.** In *Run an evaluation*, choose the validated model, the
   validated dataset and the task type, then press **Run evaluation**.
6. **Read the results.** Metrics appear right below the form. Every run is also
   added to *Evaluation history*, which shows time, model, dataset, task type and
   metrics.

Only models and datasets that passed validation appear in the evaluation
dropdowns. You can evaluate several models against one dataset, or one model
against several datasets.

## 3. Limits

| Item | Limit |
|---|---|
| Model file types | `.pkl`, `.joblib` |
| Model file size | 100 MB |
| Dataset file type | `.csv` |
| Dataset size | 50,000 rows |
| Task types | classification, regression |

## 4. Reading the metrics

**Classification** (precision, recall and F1 are weighted by class size)

| Metric | Meaning | Best |
|---|---|---|
| Accuracy | Share of rows predicted correctly | 1.0 |
| Precision | Of the rows predicted as a class, how many were right | 1.0 |
| Recall | Of the rows truly in a class, how many were found | 1.0 |
| F1 | Balance of precision and recall | 1.0 |
| ROC-AUC | How well the predicted probabilities rank the classes (0.5 is random guessing) | 1.0 |

ROC-AUC is shown as unavailable, with a note, if the model has no
`predict_proba()` or the score cannot be computed.

**Regression**

| Metric | Meaning | Best |
|---|---|---|
| MAE | Average absolute error, in the target's units | 0 |
| MSE | Average squared error, punishes big misses | 0 |
| RMSE | Square root of MSE, in the target's units | 0 |
| R² | Share of the target's variation the model explains (can be negative) | 1.0 |

## 5. Troubleshooting

| Message | Cause and fix |
|---|---|
| Unsupported model file type | Use a `.pkl` or `.joblib` file |
| Model file exceeds the 100 MB limit | Use a smaller model |
| The uploaded file is empty | The file has no content; re-save it |
| Could not load the model file | The file is corrupt, was not saved with joblib/pickle, or was saved with an incompatible scikit-learn version |
| File loaded but is not a usable model (no predict() method found) | You saved something other than the model (for example a dict); save the model itself |
| Unsupported dataset file type | Use a `.csv` file |
| A target variable must be specified | Type the target column name before uploading |
| Target variable 'x' is not a column | The name is case-sensitive; the message lists the real columns |
| Could not read the dataset file | The CSV is malformed, for example rows with different numbers of fields |
| Dataset has N rows, exceeding the 50000 row limit | Split or sample the data |
| Target variable 'x' has N missing value(s) | Fill or drop rows with an empty target |
| Dataset has no feature columns besides the target variable | The CSV needs at least one column besides the target |
| This model/dataset has not passed validation yet | Press **Validate** on it first |
| Model expects feature(s) not found in the dataset | The dataset is missing columns the model was trained on |
| Model expects N feature(s) but the dataset has M | Column count differs from training |
| Task is 'regression' but the model looks like a classifier | Choose the other task type |
| Task is 'classification' but the target looks continuous | The target has many distinct decimal values; this is probably regression |
| Evaluation failed while running the model | The model raised an error on this data; the message gives the reason, often a data type problem |
| The uploaded file is too large | The upload exceeded the server limit (110 MB) |
| Something went wrong | Unexpected server error; check the terminal where the app is running |

Other problems:

- **Page does not load:** make sure `python app.py` is still running and use
  http://127.0.0.1:5000.
- **`ModuleNotFoundError`:** activate the virtual environment, then run
  `pip install -r requirements.txt` again.
- **Port 5000 already in use:** stop the other program using it. On macOS,
  AirPlay Receiver uses port 5000; turn it off in System Settings.
- **Start over:** stop the app and delete the files inside `backend/storage/`
  (keep the `models` and `datasets` folders).

## 6. HTTP API

The page uses these endpoints, and you can call them directly. Errors always
return JSON like `{"error": "message"}`.

| Method and path | Purpose | Success |
|---|---|---|
| `GET /api/models` | List uploaded models | 200 |
| `POST /api/models/upload` | Upload a model (form field `model_file`) | 201 |
| `POST /api/models/<id>/validate` | Validate a model | 200 |
| `GET /api/datasets` | List uploaded datasets | 200 |
| `POST /api/datasets/upload` | Upload a dataset (form fields `dataset_file`, `target_variable`) | 201 |
| `POST /api/datasets/<id>/validate` | Validate a dataset | 200 |
| `GET /api/evaluations` | List past evaluations | 200 |
| `POST /api/evaluations` | Run an evaluation (JSON: `model_id`, `dataset_id`, `task_type`) | 201 |

Status codes for errors: 400 bad input, 404 unknown id or route, 405 wrong
method, 413 file too large, 500 unexpected error.

Example with `curl`:

```bash
curl -F "model_file=@classifier.joblib" http://127.0.0.1:5000/api/models/upload
curl -X POST http://127.0.0.1:5000/api/models/<model-id>/validate

curl -F "dataset_file=@test_data.csv" -F "target_variable=target" \
     http://127.0.0.1:5000/api/datasets/upload
curl -X POST http://127.0.0.1:5000/api/datasets/<dataset-id>/validate

curl -X POST http://127.0.0.1:5000/api/evaluations \
     -H "Content-Type: application/json" \
     -d '{"model_id": "<model-id>", "dataset_id": "<dataset-id>", "task_type": "classification"}'
```
