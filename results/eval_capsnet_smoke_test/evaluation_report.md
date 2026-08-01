# Evaluation report

| Run info | |
|---|---|
| model | aeromind_capsnet |
| protocol | subject_dependent |
| fold | subject_dependent |
| checkpoint | runs\synthetic_smoke_test_capsnet\subject_dependent\best.ckpt |
| n_test_sequences | 168 |
| config_hash | 3da50b119895 |

### Workload classification (3-class)

| Metric | Value |
|---|---|
| Accuracy | 0.345 |
| Macro-F1 | 0.181 |
| Cohen's Kappa | 0.002 |
| ROC-AUC (macro, OvR) | 0.447 |
| Expected Calibration Error | 0.001 |

**Per-class F1:**

| low | medium | high |
|---|---|---|
| 0.028 | 0.516 | 0.000 |

**Confusion matrix** (rows = actual, columns = predicted):

| actual \ predicted | low | medium | high |
|---|---|---|---|
| **low** | 1 | 67 | 2 |
| **medium** | 0 | 57 | 1 |
| **high** | 1 | 39 | 0 |


### Fatigue classification (binary)

| Metric | Value |
|---|---|
| Accuracy | 0.595 |
| Macro-F1 | 0.592 |
| Cohen's Kappa | 0.204 |
| ROC-AUC (macro, OvR) | 0.678 |
| Expected Calibration Error | 0.047 |

**Per-class F1:**

| alert | fatigued |
|---|---|
| 0.558 | 0.626 |

**Confusion matrix** (rows = actual, columns = predicted):

| actual \ predicted | alert | fatigued |
|---|---|---|
| **alert** | 43 | 47 |
| **fatigued** | 21 | 57 |

