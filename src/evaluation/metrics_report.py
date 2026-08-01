"""Markdown evaluation report generator (README §11, roadmap step 80/84):
confusion matrix, ROC-AUC, and calibration (ECE) tables from a
`ClassificationReport`."""

from __future__ import annotations

from src.data.synthetic import FATIGUE_LABELS, WORKLOAD_LABELS
from src.utils.metrics import ClassificationReport

_LABEL_SETS = {2: FATIGUE_LABELS, 3: WORKLOAD_LABELS}


def _confusion_matrix_markdown(report: ClassificationReport, labels: tuple[str, ...]) -> str:
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = []
    for i, row_label in enumerate(labels):
        row_vals = " | ".join(str(v) for v in report.confusion[i])
        rows.append(f"| **{row_label}** | {row_vals} |")
    return "\n".join([header, sep, *rows])


def _per_class_f1_markdown(report: ClassificationReport, labels: tuple[str, ...]) -> str:
    header = "| " + " | ".join(labels) + " |"
    sep = "|---" * len(labels) + "|"
    row = "| " + " | ".join(f"{v:.3f}" for v in report.per_class_f1) + " |"
    return "\n".join([header, sep, row])


def _task_section(title: str, report: ClassificationReport, n_classes: int) -> str:
    labels = _LABEL_SETS.get(n_classes, tuple(f"class_{i}" for i in range(n_classes)))
    roc_auc_str = f"{report.roc_auc:.3f}" if report.roc_auc is not None else "n/a"
    ece_str = f"{report.ece:.3f}" if report.ece is not None else "n/a"
    return f"""### {title}

| Metric | Value |
|---|---|
| Accuracy | {report.accuracy:.3f} |
| Macro-F1 | {report.macro_f1:.3f} |
| Cohen's Kappa | {report.kappa:.3f} |
| ROC-AUC (macro, OvR) | {roc_auc_str} |
| Expected Calibration Error | {ece_str} |

**Per-class F1:**

{_per_class_f1_markdown(report, labels)}

**Confusion matrix** (rows = actual, columns = predicted):

{_confusion_matrix_markdown(report, labels)}
"""


def generate_report(
    workload_report: ClassificationReport,
    fatigue_report: ClassificationReport,
    meta: dict,
) -> str:
    """Full markdown evaluation report for one checkpoint/fold, combining
    the workload and fatigue classification reports with run metadata."""
    meta_rows = "\n".join(f"| {k} | {v} |" for k, v in meta.items())
    return f"""# Evaluation report

| Run info | |
|---|---|
{meta_rows}

{_task_section("Workload classification (3-class)", workload_report, n_classes=3)}

{_task_section("Fatigue classification (binary)", fatigue_report, n_classes=2)}
"""
