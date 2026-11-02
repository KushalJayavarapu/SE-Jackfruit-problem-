# User Guide

How to use the Model Evaluator, prepare files for it, read the results and fix
common problems. For installation, see the [main README](../README.md).

## 1. Log in

Go to http://127.0.0.1:5000.

- **New Analyst:** choose **Create an Analyst account**, enter your name, email
  and a password of at least 8 characters, and you are signed in straight away.
- **Returning user:** log in with your email and password. Analysts and Admins
  use the same page. Admins land on the Accounts page, Analysts on the Overview.
- **Log out** is at the top right.
- Sessions last 8 hours.

If the message says your account is suspended, ask an Admin to reactivate it.
There is no password reset yet. An Admin can create a new account for you.

## 2. Prepare your files

### The model

A trained scikit-learn style object saved as `.pkl` or `.joblib`, with a
`predict()` method. For ROC-AUC and the ROC curve it also needs `predict_proba()`.

Example that creates a compatible classifier:

```python
import joblib
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
  not matter. With a plain NumPy array the app can only check the column count.
- **Use the same scikit-learn version** to save and to load.
- **Save only the model**, not a dictionary or tuple holding it.
- If your model needs preprocessing (scaling, encoding), save the whole
  `Pipeline` so the raw dataset columns work as they are.
- Only upload model files you created or trust. Loading one runs code.

### The dataset

- A `.csv` file with a header row, up to **50,000 rows**.
- One column is the **target** (the true answer). Every other column is a feature.
- The target column must have **no empty values**.
- The feature columns must be the ones the model was trained on. Extra columns
  are allowed: they are ignored and mentioned in the result notes.
- For classification, target values must be labels the model learned. For
  regression, the target must be numeric.
- Use data the model was **not** trained on, otherwise the scores look better
  than they really are. The app can't check this for you.

Ready-made files are in `backend/sample_data/`.

## 3. Run an evaluation

1. **Models page:** choose a file and press **Upload model**, then press
   **Validate** on its card. A valid model shows its type, whether it predicts
   categories or numbers, whether it gives probabilities, and the columns it expects.
2. **Datasets page:** choose a CSV. The **Target column** list fills with its
   column names; pick the one to predict. Press **Upload dataset**, then **Validate**.
3. After validating, each card lists which of your other files it fits with, and
   if not, why.
4. **Evaluate page:** choose a validated model and dataset. A line under them
   says whether they fit together. Choose **Classification** or **Regression**
   (the model's kind is preselected) and press **Run evaluation**.
5. You land on the results page for that run.

Delete removes an uploaded file. Evaluations that used it stay in your history.

## 4. Results

The results page shows:

- **Metrics**, with whether higher or lower is better.
- **Charts** that depend on the task:
  - Classification: **confusion matrix** (rows are actual labels, columns are
    predicted labels) and **ROC curve**. The ROC curve appears only if the model
    gives probabilities. Multiclass models get one curve per class.
  - Regression: **residual plot** (error against predicted value; a good model
    scatters evenly around zero) and **predicted vs actual** (points on the
    dashed line are perfect).
- **Notes**, such as ignored dataset columns or why ROC-AUC isn't available.
- **Details**: file names, row count, run time and the evaluation ID.

Classification metrics never appear next to regression metrics, or the other way round.

### What the metrics mean

**Classification** (precision, recall and F1 are weighted by class size)

| Metric | Meaning | Best |
|---|---|---|
| Accuracy | Share of rows predicted correctly | 1.0 |
| Precision | Of the rows predicted as a class, how many were right | 1.0 |
| Recall | Of the rows truly in a class, how many were found | 1.0 |
| F1 | Balance of precision and recall | 1.0 |
| ROC-AUC | How well predicted probabilities rank the classes (0.5 is random guessing) | 1.0 |

**Regression**

| Metric | Meaning | Best |
|---|---|---|
| MAE | Average absolute error, in the target's units | 0 |
| MSE | Average squared error; punishes big misses | 0 |
| RMSE | Square root of MSE, in the target's units | 0 |
| R² | Share of the target's variation the model explains (can be negative) | 1.0 |

## 5. History, comparing and reports

- **History** lists every evaluation you ran, newest first. Click one to reopen
  it. It shows exactly the numbers and charts from when it was run.
- **Compare:** tick two or more evaluations of the **same task type** and press
  **Compare selected**. You get a table and bar charts side by side. The app
  does not pick a winner; which model is better depends on your problem. If you
  tick a classification and a regression run, you get a warning and Compare stays off.
- **Reports:** tick one or more evaluations and press **Download report**, or use
  the button on a results or compare page. The PDF contains, for each
  evaluation, the model, dataset, metrics, notes and charts, plus a comparison
  table when there are several. It is built from the saved results, so it
  matches what you saw, even if you have since deleted the files.

## 6. For Admins

Admins see **Accounts** and **Audit log** at the left of the top bar. They also
have the same Overview, Models, Datasets, Evaluate and History pages, but
see only their own data there.

**Accounts**

- Lists every account with role, status and last login.
- **Suspend** logs the person out immediately and blocks login. **Reactivate** reverses it.
- **Make Admin** and **Make Analyst** change the role at once.
- **Create an account** is the only way to add another Admin from the web.
  Share the starting password with the person directly.
- The buttons are off for your own row and for the last active Admin. This is
  enforced on the server too.

**Audit log** lists each account creation, suspension, reactivation and role
change with who did it, who it affected, and when.

**First Admin:** run `flask --app app create-admin` in the `backend` folder.

## 7. Limits

| Item | Limit |
|---|---|
| Model file types | `.pkl`, `.joblib` |
| Model file size | 100 MB |
| Dataset file type | `.csv` |
| Dataset size | 50,000 rows |
| Task types | classification, regression |
| Password | at least 8 characters |

## 8. Troubleshooting

| Message | Cause and fix |
|---|---|
| Incorrect email or password. | Check both. Emails are not case-sensitive, passwords are. |
| This account is suspended. | Ask an Admin to reactivate it. |
| An account with this email already exists. | Log in instead. |
| Your session has expired. Refresh the page and try again. | Reload the page and repeat the action. |
| Please log in to continue. | Your session ended; log in again. |
| This area is only for Admins. | The page needs the Admin role. |
| Unsupported model file type | Use a `.pkl` or `.joblib` file. |
| Model file exceeds the 100 MB limit | Use a smaller model. |
| The uploaded file is empty | The file has no content; save it again. |
| Could not load the model file | The file is corrupt, was not saved with joblib/pickle, or came from an incompatible scikit-learn version. |
| File loaded but is not a usable model (no predict() method found) | You saved something other than the model (for example a dict). Save the model itself. |
| Unsupported dataset file type | Use a `.csv` file. |
| A target variable must be specified | Choose the target column before uploading. |
| Target variable 'x' is not a column | The name must match a column exactly. The message lists the real columns. |
| Could not read the dataset file | The CSV is malformed, for example rows with different numbers of fields. |
| Dataset has N rows, exceeding the 50000 row limit | Split or sample the data. |
| Target variable 'x' has N missing value(s) | Fill or drop rows with an empty target. |
| Dataset has no feature columns besides the target variable | The CSV needs at least one column besides the target. |
| This model/dataset has not passed validation yet | Press **Validate** on it first. |
| Model expects feature(s) not found in the dataset | The dataset is missing columns the model was trained on. |
| Model expects N feature(s) but the dataset has M | The column count differs from training. |
| Task is 'regression' but the model looks like a classifier | Choose the other task type. |
| Task is 'classification' but the model looks like a regressor | Choose the other task type. |
| Task is 'classification' but the target looks continuous | The target has many distinct decimal values, so this is probably regression. |
| Task is 'regression' but the target column is not numeric | A regression target must be numbers. |
| Evaluation failed while running the model | The model raised an error on this data; the message gives the reason, often a data type problem. Nothing is saved. |
| The selected evaluations have different task types | Compare only classification with classification, or regression with regression. |
| One or more of the selected evaluations no longer exist | Reload History and select again. |
| You cannot change your own account / role | Ask another Admin. |
| This is the last active Admin | Make another account an active Admin first. |
| The uploaded file is too large | The upload exceeded the server limit (110 MB). |
| Something went wrong | Unexpected server error; check the terminal where the app is running. |

Other problems:

- **Page does not load:** make sure `python app.py` is still running and use
  http://127.0.0.1:5000.
- **`ModuleNotFoundError`:** activate the virtual environment, then run
  `pip install -r requirements.txt` again.
- **Port 5000 already in use:** stop the other program using it. On macOS,
  AirPlay Receiver uses port 5000; turn it off in System Settings.
- **Locked out of the only Admin account:** run
  `flask --app app create-admin` to create another one.
- **Start over:** stop the app and delete everything inside `backend/storage/`
  except the `models` and `datasets` folders (and the `.gitkeep` files in them).

## 9. HTTP API

The pages use these endpoints, and you can call them yourself. All need a
logged-in session cookie, and changing requests need the CSRF token (see
`backend/README.md`). Errors are JSON like `{"error": "message"}`.

| Method and path | Purpose | Success |
|---|---|---|
| `GET /api/me` | The signed-in user | 200 |
| `GET /api/models` | Your models | 200 |
| `POST /api/models/upload` | Upload a model (form field `model_file`) | 201 |
| `POST /api/models/<id>/validate` | Validate; response includes `compatibility` | 200 |
| `DELETE /api/models/<id>` | Delete a model | 200 |
| `GET /api/datasets` | Your datasets | 200 |
| `POST /api/datasets/upload` | Upload (form fields `dataset_file`, `target_variable`) | 201 |
| `POST /api/datasets/<id>/validate` | Validate; response includes `compatibility` | 200 |
| `DELETE /api/datasets/<id>` | Delete a dataset | 200 |
| `GET /api/evaluations` | Your evaluations, newest first | 200 |
| `POST /api/evaluations` | Run one (JSON: `model_id`, `dataset_id`, `task_type`) | 201 |
| `GET /api/evaluations/<id>` | Reopen a saved evaluation | 200 |
| `GET /api/evaluations/<id>/charts/<kind>.png` | A chart image | 200 |
| `GET /api/compare?ids=a,b` | Side-by-side metrics | 200 |
| `GET /api/compare/chart.png?ids=a,b` | Comparison bar charts | 200 |
| `GET /api/reports?ids=a,b` | PDF report | 200 |
| `GET /api/admin/users` | All accounts (Admin) | 200 |
| `POST /api/admin/users` | Create an account (Admin) | 201 |
| `POST /api/admin/users/<id>/status` | `{"status": "active" or "suspended"}` (Admin) | 200 |
| `POST /api/admin/users/<id>/role` | `{"role": "analyst" or "admin"}` (Admin) | 200 |
| `GET /api/admin/audit` | Audit log (Admin) | 200 |

Chart kinds: `confusion_matrix`, `roc_curve`, `residuals`, `predicted_vs_actual`.

Error status codes: 400 bad input or a failed check, 401 not logged in, 403 not
allowed, 404 not found (including other people's records), 405 wrong method,
413 file too large, 500 unexpected error.
