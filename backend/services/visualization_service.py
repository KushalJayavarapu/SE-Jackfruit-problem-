"""
FR-05 / UC-08: charts for an evaluation.

build_chart_data() runs once, at evaluation time, and its output is saved with
the evaluation. render() draws a PNG from that saved data only, so a reopened
evaluation or a report always shows the same charts as the original run (FR-07).

Charts per task:
  classification -> confusion matrix, plus ROC curve when ROC-AUC was computed
  regression     -> residual plot, predicted vs actual
"""
import io

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

import config  # noqa: E402
from services.metrics import METRIC_INFO  # noqa: E402

INK = "#17213a"
MUTED = "#5b6577"
GRID = "#dde2ea"
ACCENT = "#2f45c5"
SERIES = ["#2f45c5", "#d1495b", "#00916e", "#e09f1f", "#6b4e9b",
          "#3c91e6", "#8d6a3f", "#c05299", "#2b7a78", "#5f6b7a"]
MAX_CONFUSION_LABELS = 30

CHART_TITLES = {
    "confusion_matrix": "Confusion matrix",
    "roc_curve": "ROC curve",
    "residuals": "Residual plot",
    "predicted_vs_actual": "Predicted vs actual",
}


# ---------------------------------------------------------------- chart data

def _thin(values, limit):
    """Keep at most `limit` evenly spaced points, always keeping both ends."""
    values = np.asarray(values)
    if len(values) <= limit:
        return values
    idx = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(int))
    return values[idx]


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value if isinstance(value, (int, float, str, bool)) else str(value)


def _classification_data(y, y_pred, proba, classes, roc_available):
    data = {}
    labels = list(pd.unique(pd.Series(list(classes) + list(pd.unique(y)) + list(pd.unique(y_pred)))))
    if len(labels) <= MAX_CONFUSION_LABELS:
        matrix = skm.confusion_matrix(y, y_pred, labels=labels)
        data["confusion_matrix"] = {
            "labels": [str(_jsonable(v)) for v in labels],
            "matrix": matrix.astype(int).tolist(),
        }

    if roc_available and proba is not None:
        curves = []
        y_arr = np.asarray(y)
        targets = [(classes[1], proba[:, 1])] if len(classes) == 2 else \
            [(c, proba[:, i]) for i, c in enumerate(classes[:config.MAX_ROC_CLASSES])]
        for label, scores in targets:
            positives = (y_arr == label)
            if positives.all() or not positives.any():
                continue  # curve is undefined when only one class is present
            fpr, tpr, _ = skm.roc_curve(positives.astype(int), scores)
            curves.append({
                "label": str(_jsonable(label)),
                "auc": float(skm.auc(fpr, tpr)),
                "fpr": [round(float(v), 5) for v in _thin(fpr, config.MAX_ROC_POINTS)],
                "tpr": [round(float(v), 5) for v in _thin(tpr, config.MAX_ROC_POINTS)],
            })
        if curves:
            data["roc_curve"] = {"binary": len(classes) == 2, "curves": curves}
    return data


def _regression_data(y, y_pred):
    actual = np.asarray(y, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    n = len(actual)
    if n > config.MAX_RESIDUAL_POINTS:
        idx = np.sort(np.random.default_rng(0).choice(n, config.MAX_RESIDUAL_POINTS, replace=False))
        actual, predicted = actual[idx], predicted[idx]
    points = {
        "actual": [round(float(v), 6) for v in actual],
        "predicted": [round(float(v), 6) for v in predicted],
        "sampled": n > config.MAX_RESIDUAL_POINTS,
        "total_points": n,
    }
    return {"residuals": points, "predicted_vs_actual": points}


def build_chart_data(task_type, y, y_pred, proba, classes, roc_available):
    if task_type == "classification":
        return _classification_data(y, y_pred, proba, classes, roc_available)
    return _regression_data(y, y_pred)


def available_charts(chart_data):
    order = ["confusion_matrix", "roc_curve", "residuals", "predicted_vs_actual"]
    return [{"kind": k, "title": CHART_TITLES[k]} for k in order if k in (chart_data or {})]


# ---------------------------------------------------------------- rendering

def _figure(width=6.4, height=4.6):
    fig = Figure(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor("white")
    return fig


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def _draw_confusion(data):
    labels, matrix = data["labels"], np.array(data["matrix"])
    n = len(labels)
    size = max(4.2, min(9, 0.45 * n + 3))
    fig = _figure(size + 0.8, size)
    ax = fig.add_subplot()
    cmap = LinearSegmentedColormap.from_list("ink", ["#ffffff", "#c9d1f3", ACCENT, "#16236b"])
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(n), labels, rotation=45 if n > 4 else 0, ha="right" if n > 4 else "center")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n <= 15:
        threshold = matrix.max() / 2 if matrix.max() else 1
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10 if n < 8 else 8,
                        color="white" if matrix[i, j] > threshold else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    return fig


def _draw_roc(data):
    fig = _figure(5.6, 5.0)
    ax = fig.add_subplot()
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa3b2", linewidth=1, label="Random guess (AUC 0.50)")
    for i, curve in enumerate(data["curves"]):
        name = f"AUC {curve['auc']:.3f}" if data["binary"] else f"Class {curve['label']} (AUC {curve['auc']:.3f})"
        ax.plot(curve["fpr"], curve["tpr"], color=SERIES[i % len(SERIES)], linewidth=2, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    return fig


def _draw_residuals(data):
    actual, predicted = np.array(data["actual"]), np.array(data["predicted"])
    fig = _figure()
    ax = fig.add_subplot()
    _style(ax)
    ax.axhline(0, color="#9aa3b2", linestyle="--", linewidth=1)
    ax.scatter(predicted, actual - predicted, s=12, alpha=0.55, color=ACCENT, edgecolors="none")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual − predicted)")
    if data.get("sampled"):
        ax.set_title(f"Random sample of {len(actual):,} of {data['total_points']:,} rows",
                     fontsize=9, color=MUTED, loc="left")
    return fig


def _draw_predicted_vs_actual(data):
    actual, predicted = np.array(data["actual"]), np.array(data["predicted"])
    fig = _figure(5.6, 5.0)
    ax = fig.add_subplot()
    _style(ax)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#9aa3b2", linewidth=1, label="Perfect prediction")
    ax.scatter(actual, predicted, s=12, alpha=0.55, color=ACCENT, edgecolors="none")
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    return fig


DRAWERS = {
    "confusion_matrix": _draw_confusion,
    "roc_curve": _draw_roc,
    "residuals": _draw_residuals,
    "predicted_vs_actual": _draw_predicted_vs_actual,
}


def render(kind, chart_data):
    """PNG bytes for one saved chart, or None if this evaluation has no such chart."""
    if kind not in DRAWERS or kind not in (chart_data or {}):
        return None
    return _png(DRAWERS[kind](chart_data[kind]))


def render_comparison(evaluations, metric_keys, numbers=None):
    """One small bar chart per metric, one bar per evaluation (FR-06)."""
    keys = [k for k in metric_keys if any(e["metrics"].get(k) is not None for e in evaluations)]
    n = len(evaluations)
    numbers = numbers or list(range(1, n + 1))
    fig = _figure(6.8, len(keys) * (0.42 * n + 0.75) + 0.3)
    labels = [f"{num}. {e['model_filename'][:28]}" for num, e in zip(numbers, evaluations)]
    for row, key in enumerate(keys):
        ax = fig.add_subplot(len(keys), 1, row + 1)
        _style(ax)
        ax.grid(axis="y", visible=False)
        values = [e["metrics"].get(key) for e in evaluations]
        shown = [v if v is not None else 0 for v in values]
        bars = ax.barh(range(n), shown, color=[SERIES[i % len(SERIES)] for i in range(n)], height=0.6)
        ax.set_yticks(range(n), labels)
        ax.invert_yaxis()
        name, higher_better = METRIC_INFO[key]
        ax.set_title(f"{name} ({'higher' if higher_better else 'lower'} is better)",
                     fontsize=10, color=INK, loc="left")
        top = max([abs(v) for v in shown] + [1e-9])
        for bar, value in zip(bars, values):
            text = "n/a" if value is None else f"{value:.4g}"
            ax.text(bar.get_width() + top * 0.01 if bar.get_width() >= 0 else top * 0.01,
                    bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=8.5, color=INK)
        ax.set_xlim(min(0, min(shown) * 1.15), top * 1.18)
    fig.tight_layout()
    return _png(fig)
