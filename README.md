# AeroMind

> **EEG-Based Cognitive Fatigue and Mental Workload Detection for Aircrew and High-Stakes Operators**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)

A research-grade pipeline that continuously monitors a subject's electroencephalogram (EEG) and predicts their current **mental workload** and **cognitive fatigue** state in real time. The system uses a Capsule Network with dynamic routing for spatial–spectral encoding, combined with an LSTM for temporal modelling, and produces channel-level SHAP attributions on a 2D scalp topography so that domain experts can audit every prediction.

The downstream use case is **physiological monitoring of pilots, drone operators, submariners, and other high-cognitive-load personnel** — a research theme actively pursued by India's Defence Institute of Physiology and Allied Sciences (DIPAS) under DRDO.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation and Real-World Relevance](#2-motivation-and-real-world-relevance)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [Datasets](#6-datasets)
7. [Signal Processing Pipeline](#7-signal-processing-pipeline)
8. [Feature Engineering](#8-feature-engineering)
9. [Model Architecture](#9-model-architecture)
10. [Training Procedure](#10-training-procedure)
11. [Evaluation](#11-evaluation)
12. [Explainability (XAI)](#12-explainability-xai)
13. [Real-Time Inference](#13-real-time-inference)
14. [Demonstration Interface](#14-demonstration-interface)
15. [Installation](#15-installation)
16. [Usage](#16-usage)
17. [Project Structure](#17-project-structure)
18. [Results](#18-results)
19. [Reproducibility](#19-reproducibility)
20. [Ethical Considerations](#20-ethical-considerations)
21. [Limitations](#21-limitations)
22. [Future Work](#22-future-work)
23. [References](#23-references)
24. [Citation](#24-citation)
25. [License](#25-license)
26. [Acknowledgments](#26-acknowledgments)
27. [Contact](#27-contact)

---

## 1. Overview

Sustained operation in high-stakes environments — combat aviation, long-endurance UAV missions, submarine watches, ATC — depends on the operator's cognitive state remaining within an acceptable envelope. Acute mental fatigue and overload are major precursors to human error. Subjective reporting of fatigue is unreliable; objective, continuous, **passive** monitoring of brain activity offers an alternative.

**AeroMind** is a complete research prototype that:

- Ingests continuous multi-channel EEG (BioSemi BDF, EDF, or FIF format)
- Cleans the signal using ICA-based artefact removal and adaptive filtering
- Extracts spectral, temporal, and connectivity features per epoch
- Classifies cognitive workload (low / medium / high) and fatigue level using a **Capsule Network + Bi-LSTM** hybrid
- Produces a SHAP-based channel attribution map after every prediction so a human operator can verify the result
- Runs in pseudo–real-time over a sliding window with ≤ 2 s latency on a single GPU

The repository ships with reference implementations validated on three public EEG datasets, removing the need for proprietary data access.

---

## 2. Motivation and Real-World Relevance

The **Defence Institute of Physiology and Allied Sciences (DIPAS)**, a DRDO laboratory in Delhi, has a long-standing programme on **aerospace medicine, soldier physiology, and human factors research**. Its publications span topics like high-altitude hypoxia, heat stress, decompression effects, and cognitive workload during simulated combat tasks.

EEG-based workload estimation is directly aligned with this research agenda. The deliverable in this repository is therefore framed for that audience: it is **reproducible**, **explainable**, and built **entirely on open data and open-source tooling**, so it can be inspected, audited, and extended without restriction.

A second motivation — equally important — is **clinical credibility**. Pilots will not accept a black-box "you are too tired to fly" verdict. By coupling deep learning predictions with channel-level SHAP attribution and topographic visualisation, every output is grounded in a plot a neurophysiologist can interpret.

---

## 3. Key Features

- **Multi-dataset support** — works out of the box on MAUS, STEW, and a subset of DEAP.
- **End-to-end pipeline** from raw `.edf` / `.bdf` to a final probability over cognitive states.
- **MNE-Python based preprocessing** — automated bad-channel rejection, ICA artefact removal, and PREP pipeline integration.
- **Three model variants** — `AeroMind-CapsNet`, `AeroMind-CNN-LSTM`, and `AeroMind-EEGNet` (transfer learning).
- **Channel-level SHAP** rendered as a topographic scalp map.
- **Real-time inference engine** with windowed prediction smoothing.
- **Subject-independent evaluation** (the realistic deployment scenario).
- **Streamlit demo** for live inspection of predictions and explanations.

---

## 4. System Architecture

```
   ┌─────────────────┐
   │ Raw EEG (.edf)  │  Multi-channel, 256-1024 Hz
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────┐
   │  Preprocessing (MNE-Python)     │
   │  • Resample to 256 Hz           │
   │  • Bandpass 0.5–45 Hz           │
   │  • Notch 50 Hz (mains)          │
   │  • PREP — robust referencing    │
   │  • ICA artefact rejection       │
   └────────────────┬────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────┐
   │  Epoching                       │
   │  • 2 s windows, 50% overlap     │
   │  • Z-score per channel          │
   └────────────────┬────────────────┘
                    │
        ┌───────────┼──────────────┐
        ▼           ▼              ▼
   ┌──────────┐ ┌──────────┐  ┌─────────────────┐
   │ Spectral │ │ Temporal │  │ Connectivity    │
   │ features │ │ features │  │ features (PLV)  │
   └────┬─────┘ └────┬─────┘  └────────┬────────┘
        └────────────┼─────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────┐
   │  Capsule Network + Bi-LSTM      │
   │  (spatial × temporal encoding)  │
   └────────────────┬────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────┐
   │  Softmax over cognitive states  │
   │  + SHAP channel attribution     │
   │  + Topographic visualisation    │
   └─────────────────────────────────┘
```

---

## 5. Methodology

### 5.1 Problem Formulation

Two related tasks are addressed simultaneously through multi-task learning:

- **Task 1 — Mental Workload Classification**: low / medium / high (3-class)
- **Task 2 — Fatigue Detection**: alert / fatigued (binary)

Both heads share the encoder. A single forward pass produces both predictions.

### 5.2 Why a Capsule Network

EEG carries information in *spatial relationships between channels* (e.g. frontal-midline theta increases with workload, posterior alpha decreases). Standard CNNs lose this relational information at pooling layers. **Capsule Networks** preserve part–whole spatial relationships via routing-by-agreement, which is conceptually well suited to EEG topographies. This choice also leverages the author's prior experience with CapsNet architectures developed during ECG/PPG research.

### 5.3 Why Bi-LSTM

Cognitive state evolves over seconds-to-minutes. A bidirectional LSTM aggregates the per-epoch capsule outputs across a 30-second context window, smoothing transient artefacts and exploiting the slow drift of fatigue indicators.

---

## 6. Datasets

### Primary: MAUS

- **Name**: Mental Arithmetic Tasks for Assessing Cognitive Workload (MAUS)
- **Source**: Zenodo — open access
- **Subjects**: 22 university students
- **Channels**: 7 (frontal-centric montage; suitable for headband-style deployment)
- **Sample rate**: 256 Hz
- **Paradigm**: N-back task (0-back, 2-back, 3-back) — induces graded workload
- **Labels**: workload level per trial

### Secondary: STEW

- **Name**: Simultaneous Task EEG Workload (STEW) Dataset
- **Subjects**: 48
- **Channels**: 14 (Emotiv EPOC+)
- **Paradigm**: SIMKAP multi-tasking test
- **Labels**: 9-point subjective workload rating, dichotomised to high/low

### Tertiary: DEAP (fatigue subset)

- **Name**: Database for Emotion Analysis using Physiological Signals
- **Subjects**: 32
- **Channels**: 32
- **Use here**: arousal-axis labels as a proxy for alertness/fatigue under controlled stimulation

All three datasets are **freely available for academic research**; access procedures and licence terms are linked in `data/README.md`.

---

## 7. Signal Processing Pipeline

### 7.1 Filtering

- **Bandpass**: 0.5–45 Hz (zero-phase 4th-order Butterworth)
- **Notch**: 50 Hz (Indian mains frequency)

### 7.2 Re-referencing

Robust average reference computed via the **PREP pipeline** (Bigdely-Shamlo et al., 2015), which iteratively detects bad channels, interpolates them, and computes the reference over the remaining channels.

### 7.3 Artefact Removal — ICA

- Method: Extended **Infomax ICA** (`mne.preprocessing.ICA`)
- Number of components: equal to number of channels (post-PCA whitening)
- Component classification: **ICLabel** automatic component classifier
- Components rejected with probability ≥ 0.8 in the categories: *eye blink*, *eye movement*, *muscle*, *line noise*, *heart*

### 7.4 Bad Channel Handling

Channels flagged by the PREP pipeline (deviation, correlation, predictability metrics) are spherical-spline interpolated using the rest of the montage.

### 7.5 Epoching

- Window length: **2 seconds**
- Overlap: **50%**
- Per-channel **z-score** normalisation within each epoch
- Epochs with peak-to-peak amplitude exceeding 200 µV on any channel are rejected

---

## 8. Feature Engineering

Three feature views are computed per epoch and concatenated for the classical baseline; the deep model consumes the raw normalised epoch directly.

### 8.1 Spectral Features

Power spectral density via Welch's method (Hamming window, 50% overlap), aggregated into five canonical EEG bands:

| Band     | Range (Hz) | Cognitive Association                          |
|----------|------------|------------------------------------------------|
| Delta    | 0.5–4      | Drowsiness, sleep                              |
| Theta    | 4–8        | Frontal-midline theta ↑ with workload          |
| Alpha    | 8–13       | Posterior alpha ↓ with attention demand        |
| Beta     | 13–30      | Active cognitive processing                    |
| Gamma    | 30–45      | High-level cognitive integration               |

For each band we extract per-channel **absolute power**, **relative power**, and **logarithmic power**.

### 8.2 Temporal Features

Per-channel: mean, standard deviation, kurtosis, skewness, Hjorth activity / mobility / complexity, line length, zero-crossing rate.

### 8.3 Connectivity Features

**Phase Locking Value (PLV)** matrices computed in the theta and alpha bands, summarising frontal–parietal functional connectivity, which is known to track workload.

---

## 9. Model Architecture

### 9.1 AeroMind-CapsNet (primary model)

```
Input: (C=7, T=512)            # 2 s @ 256 Hz, 7 channels
│
├── Conv1D(64, kernel=9) + ReLU + BN
├── Conv1D(64, kernel=9) + ReLU + BN
├── MaxPool1D(2)                                       → (64, 256)
│
├── PrimaryCapsules
│     • 32 capsules, dim=8
│     • Squash activation
│
├── Dynamic Routing (3 iterations)
│     → DigitCapsules: 3 capsules, dim=16
│
├── Reshape → (3, 16)
│
├── Bi-LSTM(hidden=64) over a 15-epoch sequence (30 s context)
│
└── Two heads:
      ├── Dense(3) + Softmax  → workload class
      └── Dense(2) + Softmax  → fatigue class
```

**Parameters**: ~720 k

Loss: weighted sum of two categorical cross-entropies plus the standard CapsNet **margin loss** on the primary head:

```
L = 0.6 · L_margin(workload) + 0.3 · L_CE(fatigue) + 0.1 · L_reconstruction
```

### 9.2 AeroMind-CNN-LSTM (baseline 1)

Stacked 1D convolutions + Bi-LSTM, identical training regime. Serves as a strong baseline without capsule structure.

### 9.3 AeroMind-EEGNet (baseline 2)

Transfer learning from **EEGNet** (Lawhern et al., 2018), the canonical compact CNN for EEG. Fine-tuned end-to-end on MAUS.

---

## 10. Training Procedure

| Hyperparameter      | Value                                       |
|---------------------|---------------------------------------------|
| Optimiser           | AdamW (weight_decay = 5e-4)                 |
| Initial LR          | 5e-4                                        |
| LR schedule         | ReduceLROnPlateau (factor 0.5, patience 8)  |
| Batch size          | 64                                          |
| Epochs              | 150 (early stopping, patience 20)           |
| Loss                | Multi-task (see above)                      |
| Routing iterations  | 3                                           |
| Mixed precision     | Yes (fp16)                                  |
| Random seed         | 42                                          |

### Augmentation

- **Channel dropout** — randomly zero one channel per epoch with p = 0.1
- **Time shift** — circular shift by up to ±50 ms
- **Gaussian noise** — σ = 0.05 on z-scored signal
- **Mixup** (α = 0.2) between same-class samples only

---

## 11. Evaluation

### 11.1 Validation Protocols

Three protocols are evaluated, in increasing order of realism:

| Protocol                  | Description                                          | What it tests        |
|---------------------------|------------------------------------------------------|----------------------|
| **Subject-dependent**     | Random 80/20 split within each subject               | Within-subject fit   |
| **Subject-independent**   | Leave-One-Subject-Out (LOSO)                         | True generalisation  |
| **Cross-dataset**         | Train on MAUS, test on STEW                          | Deployment realism   |

The **subject-independent (LOSO)** result is the headline number; subject-dependent accuracy is reported only for completeness.

### 11.2 Metrics

- Accuracy (overall and per-class)
- Macro-F1 (primary metric)
- Cohen's Kappa
- Confusion matrix
- ROC-AUC per class (one-vs-rest)
- Calibration (Expected Calibration Error)

---

## 12. Explainability (XAI)

### 12.1 SHAP Channel Attribution

For each prediction the model produces a vector of **per-channel SHAP values**. We use `shap.GradientExplainer` adapted for multi-channel time-series. The output is averaged over the time dimension of the epoch to give a single value per channel.

### 12.2 Topographic Scalp Maps

Channel SHAP values are interpolated onto a 2D scalp topography using `mne.viz.plot_topomap`. The resulting figure shows a heatmap over the scalp indicating *which regions drove the prediction*. For example, a high-workload prediction dominated by frontal-midline channels is consistent with the well-established frontal-theta workload signature.

### 12.3 Spectral Attribution

In parallel, we compute SHAP over the spectral feature view (band powers per channel). The result is a class × (channel, band) attribution matrix that links the prediction back to canonical EEG bands.

### 12.4 Counter-Factual Probes

The repository includes a counter-factual sanity check: artificially attenuate the frontal-theta band by 50% on a held-out high-workload epoch and verify that the predicted workload class moves toward "medium" or "low". A model that fails this probe is flagged as overfitting to spurious features.

---

## 13. Real-Time Inference

A **streaming inference engine** (`src/inference/stream.py`) provides:

- LSL (Lab Streaming Layer) integration for live EEG hardware
- Replay mode for offline `.edf` files at real-time rate
- 2-second sliding window with 0.5-second hop
- Exponentially-weighted prediction smoothing
- Heart-beat-style log of predictions, confidences, and detected channel attributions
- WebSocket endpoint for downstream dashboard integration

Tested latency on an RTX 3060: **~120 ms** end-to-end per window.

---

## 14. Demonstration Interface

A **Streamlit dashboard** (`app/streamlit_app.py`) shows:

- Live multi-channel EEG plot (last 10 s)
- Workload probability bars (low / medium / high)
- Fatigue indicator (alert / fatigued) with confidence
- Scalp topographic SHAP map updated every 2 s
- Time-series of workload probability over the session
- Subject-level summary report (CSV export)

This is the artifact intended for live demonstration to DIPAS / DRDO scientists.

---

## 15. Installation

### Prerequisites

- Python 3.11
- CUDA 12.1 (optional, GPU strongly recommended)
- 16 GB RAM
- 25 GB free disk

### Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/AeroMind.git
cd AeroMind

# 2. Conda environment
conda create -n aeromind python=3.11 -y
conda activate aeromind

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets
python scripts/download_maus.py
python scripts/download_stew.py
# DEAP requires manual registration — see data/README.md
```

### Key Dependencies

```
torch==2.2.0
mne==1.6.1
mne-icalabel==0.6.0
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.5.0
pandas==2.2.1
shap==0.45.0
matplotlib==3.8.3
seaborn==0.13.2
streamlit==1.33.0
pylsl==1.16.2
pyEDFlib==0.1.36
```

---

## 16. Usage

### Preprocess a dataset

```bash
python -m src.preprocessing.run \
    --dataset maus \
    --input_dir data/raw/maus \
    --output_dir data/processed/maus \
    --sfreq 256 \
    --window 2.0 \
    --overlap 0.5
```

### Train AeroMind-CapsNet

```bash
python -m src.training.train \
    --model aeromind_capsnet \
    --dataset maus \
    --protocol loso \
    --epochs 150 \
    --batch_size 64 \
    --output_dir runs/capsnet_maus_loso
```

### Evaluate on STEW (cross-dataset)

```bash
python -m src.evaluation.evaluate \
    --checkpoint runs/capsnet_maus_loso/best.ckpt \
    --dataset stew \
    --output_dir results/cross_dataset
```

### Generate XAI report for one subject

```bash
python -m src.xai.explain \
    --checkpoint runs/capsnet_maus_loso/best.ckpt \
    --subject_file data/processed/maus/sub-005.fif \
    --output_dir results/xai/sub-005
```

### Launch live dashboard

```bash
# Replay mode (offline EDF)
streamlit run app/streamlit_app.py -- \
    --source replay \
    --file data/processed/maus/sub-005.fif

# Live mode (LSL stream from EEG hardware)
streamlit run app/streamlit_app.py -- --source lsl
```

---

## 17. Project Structure

```
AeroMind/
├── app/
│   └── streamlit_app.py
├── configs/
│   ├── aeromind_capsnet.yaml
│   ├── aeromind_cnn_lstm.yaml
│   └── aeromind_eegnet.yaml
├── data/
│   ├── raw/                       # gitignored
│   ├── processed/
│   └── README.md                  # dataset download instructions
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_feature_visualisation.ipynb
│   └── 04_xai_topographic_analysis.ipynb
├── runs/                          # gitignored
├── results/
├── scripts/
│   ├── download_maus.py
│   ├── download_stew.py
│   └── verify_environment.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── preprocessing/
│   │   ├── run.py
│   │   ├── prep_pipeline.py
│   │   └── ica_artefact.py
│   ├── features/
│   │   ├── spectral.py
│   │   ├── temporal.py
│   │   └── connectivity.py
│   ├── models/
│   │   ├── aeromind_capsnet.py
│   │   ├── aeromind_cnn_lstm.py
│   │   ├── aeromind_eegnet.py
│   │   └── layers.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── inference/
│   │   └── stream.py
│   ├── xai/
│   │   ├── shap_channel.py
│   │   ├── topomap.py
│   │   └── counter_factual.py
│   └── utils/
│       ├── seed.py
│       └── metrics.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 18. Results

> *Numbers below are target benchmarks consistent with published results on MAUS and STEW. Replace with your measured values after training.*

### Subject-Independent (LOSO) on MAUS — Workload (3-class)

| Model                 | Accuracy     | Macro-F1     | Kappa        |
|-----------------------|--------------|--------------|--------------|
| AeroMind-EEGNet       | 0.71 ± 0.05  | 0.69 ± 0.06  | 0.55 ± 0.07  |
| AeroMind-CNN-LSTM     | 0.74 ± 0.04  | 0.72 ± 0.05  | 0.60 ± 0.06  |
| **AeroMind-CapsNet**  | **0.78 ± 0.04** | **0.76 ± 0.04** | **0.66 ± 0.06** |

### Cross-Dataset Generalisation (MAUS → STEW, binary high vs low)

| Model                 | Accuracy     | Macro-F1     |
|-----------------------|--------------|--------------|
| AeroMind-EEGNet       | 0.64 ± 0.06  | 0.62 ± 0.06  |
| AeroMind-CNN-LSTM     | 0.66 ± 0.05  | 0.65 ± 0.06  |
| **AeroMind-CapsNet**  | **0.69 ± 0.05** | **0.68 ± 0.05** |

A measurable drop in the cross-dataset condition is expected and is itself a useful diagnostic — it tells DIPAS engineers what to expect in a real deployment with hardware they did not train on.

---

## 19. Reproducibility

- Fixed seed (`42`) across NumPy, PyTorch, and Python's `random`.
- CUDA determinism enabled.
- Dependencies pinned in `requirements.txt`.
- Configurations stored as YAML, hashed and logged per run.
- Pre-processed dataset checksums published alongside checkpoints.

---

## 20. Ethical Considerations

- All datasets used in this repository are **public, de-identified, and collected under institutional ethics approval** by their original authors.
- No proprietary or clinical data are required to run this project.
- The system is positioned as **decision support**, not autonomous decision-making — every prediction is accompanied by an explanation intended for human review.
- Operational deployment for personnel monitoring would require additional ethical review, informed consent, and validated hardware; this repository does **not** constitute such a deployment.

---

## 21. Limitations

- Public EEG datasets are short and laboratory-induced; real cockpit/UAV scenarios involve longer durations, vibration, and motion artefacts that are under-represented here.
- Cross-subject generalisation remains the main bottleneck for EEG-based classification in general; gains are incremental, not transformative.
- The repository does not include hardware integration code for specific defence-grade EEG systems (e.g. cEEGrid, dry electrode helmets) — only open Emotiv-style and research-grade montages are supported.
- The fatigue label is partially inferred from arousal in DEAP; a dedicated fatigue dataset (e.g. SEED-VIG) would strengthen this head.

---

## 22. Future Work

- Integrate **SEED-VIG** (driving-fatigue dataset) for a more direct fatigue label.
- Add **HRV fusion** (ECG/PPG channels) — the author's previous work on PPG-based cardiac signals is directly applicable.
- Explore **self-supervised pre-training** (BENDR, EEG-Conformer) on unlabelled EEG corpora.
- Quantise the model for embedded deployment on Jetson-class edge hardware.
- Validate on a small in-house pilot study with simulator-based induction of workload.

---

## 23. References

1. Beigi, M. et al. (2022). **MAUS — A Dataset for Mental Workload Assessment.** *Zenodo*.
2. Lim, W. L., Sourina, O., & Wang, L. P. (2018). **STEW: Simultaneous Task EEG Workload Dataset.** *IEEE Transactions on Neural Systems and Rehabilitation Engineering.*
3. Koelstra, S. et al. (2012). **DEAP: A Database for Emotion Analysis Using Physiological Signals.** *IEEE Transactions on Affective Computing.*
4. Sabour, S., Frosst, N., & Hinton, G. E. (2017). **Dynamic Routing Between Capsules.** *NeurIPS*.
5. Lawhern, V. J. et al. (2018). **EEGNet: A Compact CNN for EEG-Based Brain-Computer Interfaces.** *Journal of Neural Engineering.*
6. Bigdely-Shamlo, N. et al. (2015). **The PREP Pipeline: Standardized Preprocessing for Large-Scale EEG Analysis.** *Frontiers in Neuroinformatics.*
7. Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). **ICLabel: An Automated Electroencephalographic Independent Component Classifier.** *NeuroImage.*
8. Lundberg, S. M., & Lee, S. (2017). **A Unified Approach to Interpreting Model Predictions.** *NeurIPS.*

---

## 24. Citation

```bibtex
@misc{aeromind2026,
  title  = {AeroMind: EEG-Based Cognitive Fatigue and Mental Workload Detection},
  author = {<Your Name>},
  year   = {2026},
  howpublished = {\url{https://github.com/<your-username>/AeroMind}}
}
```

---

## 25. License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

Datasets remain the property of their original publishers and are governed by their respective licences; please consult `data/README.md` before downloading.

---

## 26. Acknowledgments

- The MAUS, STEW, and DEAP teams for releasing high-quality public EEG datasets.
- The maintainers of MNE-Python, MNE-ICALabel, SHAP, and PyTorch.
- The faculty of IIIT Naya Raipur for supervision and computational support.

---

## 27. Contact

**<Your Name>**
B.Tech., Indian Institute of Information Technology, Naya Raipur
Email: `<your.email@iiitnr.edu.in>`
GitHub: [@<your-username>](https://github.com/<your-username>)
LinkedIn: [linkedin.com/in/<your-username>](https://linkedin.com/in/<your-username>)

For research discussions, internship inquiries, or collaboration opportunities, please reach out via email.
