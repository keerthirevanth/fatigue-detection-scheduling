"""
================================================================================
  WORKER FATIGUE DETECTION FROM VOICE — COMPLETE ML PIPELINE (single file)
================================================================================
  Author  : Jayakrishnan (MTech IEOR, IIT Bombay)
  Goal    : Detect the fatigue level of a worker from a short voice sample.
            The model outputs a fatigue score in [0, 1] and a category:
            alert / mild_fatigue / fatigued.

  WHY DELTA FEATURES?
  -------------------
  A raw acoustic feature (e.g., pitch mean = 180 Hz) tells us nothing about
  fatigue on its own — some people naturally have flat, low voices. What
  matters is CHANGE from that worker's own rested baseline. So the ML model
  learns patterns like:
       "when Δpitch_std drops by 15 Hz AND Δspeech_rate drops by 1 syl/s
        AND Δjitter increases by 0.3%, the worker is likely fatigued."
  This removes inter-speaker variability — the biggest noise source.

  PIPELINE
  --------
  Stage 1 : Load RAVDESS audio files
  Stage 2 : Extract 80 acoustic features per file
  Stage 3 : Build personal baselines + DELTA feature vectors per speaker
             (delta = check-in features − speaker's own baseline)
  Stage 4 : Train classifiers (RandomForest, XGBoost, SVM, MLP) on deltas
  Stage 5 : Evaluate — precision, recall, F1 (recall on 'fatigued' matters most)
  Stage 6 : Save best model + scaler + baselines to disk
  Stage 7 : Inference function — takes a new .wav + worker_id, returns score

  HOW TO GET THE DATA (download once, ~215 MB)
  --------------------------------------------
  RAVDESS (24 speakers, 1440 clips):
     https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
     or Zenodo: https://zenodo.org/records/1188976  (file: Audio_Speech_Actors_01-24.zip)
     Extract so you get: data/RAVDESS/Actor_01/*.wav, Actor_02/*.wav, ...

  FOLDER LAYOUT EXPECTED
  ----------------------
      fatigue_ml_project/
      ├── fatigue_detection_ml.py   (this file)
      ├── data/
      │   └── RAVDESS/              (from Kaggle / Zenodo)
      │       ├── Actor_01/
      │       │   ├── 03-01-01-01-01-01-01.wav
      │       │   └── ...
      │       └── Actor_24/
      └── outputs/                  (created automatically)

  RUN
  ---
      pip install numpy pandas scikit-learn xgboost librosa soundfile joblib matplotlib seaborn
      python fatigue_detection_ml.py --data_dir data/RAVDESS

  PLOTS GENERATED (saved in outputs/plots/)
  -----------------------------------------
      1. label_distribution.png       — class balance across alert/mild/fatigued
      2. speaker_distribution.png     — clips per speaker
      3. sample_waveform.png          — raw audio + energy envelope for one clip
      4. feature_boxplots.png         — key features across fatigue classes
      5. baseline_vs_fatigued.png     — one speaker's baseline vs fatigued features
      6. delta_feature_heatmap.png    — average delta vector per class
      7. pca_2d_deltas.png            — 2D PCA projection of the delta dataset
      8. confusion_matrices.png       — one per trained model
      9. model_comparison.png         — bar chart of F1 & recall per model
     10. feature_importance.png       — top features (tree-based models only)
================================================================================
"""

import os
import glob
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, recall_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ------------------------------------------------------------------ CONFIG
SAMPLE_RATE   = 16000
N_MFCC        = 13
RANDOM_STATE  = 42

# Emotion-to-fatigue mapping
# Core idea: emotions with low arousal + low energy approximate fatigue,
# while high-arousal emotions approximate the "alert" state.
RAVDESS_EMOTION_MAP = {
    "01": ("neutral",   "alert"),         # neutral   -> ALERT
    "02": ("calm",      "alert"),         # calm      -> ALERT (our baseline)
    "03": ("happy",     "alert"),         # happy     -> ALERT
    "04": ("sad",       "fatigued"),      # sad       -> FATIGUED (low energy, flat)
    "05": ("angry",     "alert"),         # angry     -> ALERT (high arousal)
    "06": ("fearful",   "mild_fatigue"),  # fearful   -> MILD (unstable voice)
    "07": ("disgust",   "fatigued"),      # disgust   -> FATIGUED (low energy)
    "08": ("surprised", "alert"),         # surprised -> ALERT
}

LABEL_TO_INT = {"alert": 0, "mild_fatigue": 1, "fatigued": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
LABEL_COLORS = {"alert": "#2ecc71", "mild_fatigue": "#f39c12", "fatigued": "#e74c3c"}


# ============================================================================
#  STAGE 1 & 2 — FEATURE EXTRACTION (80 features per .wav)
# ============================================================================
def extract_features(wav_path, sr=SAMPLE_RATE):
    """Extract an 80-dim acoustic feature vector from a single .wav file."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # Clean: trim leading/trailing silence, normalize amplitude
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr * 0.5:               # skip very short clips
        return None
    y = y / (np.max(np.abs(y)) + 1e-8)

    # ---- MFCC block (52 features) ----
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta2.mean(axis=1),
    ])

    # ---- Pitch / F0 block (6 features) ----
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) == 0:
            pitch_feats = np.zeros(6)
        else:
            pitch_feats = np.array([
                np.mean(f0v), np.std(f0v), np.min(f0v), np.max(f0v),
                np.max(f0v) - np.min(f0v),
                np.sum(voiced_flag) / max(len(voiced_flag), 1),
            ])
    except Exception:
        f0v = np.array([])
        pitch_feats = np.zeros(6)

    # ---- Voice quality block (5 features): jitter, shimmer, HNR ----
    if len(f0v) >= 3:
        periods = 1.0 / (f0v + 1e-8)
        jitter_local = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-8)
        jitter_rap   = (np.mean(np.abs(np.diff(periods, n=2)))
                        / (np.mean(periods) + 1e-8)) if len(periods) >= 3 else 0.0
    else:
        jitter_local = jitter_rap = 0.0

    rms = librosa.feature.rms(y=y)[0]
    shimmer_local = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
    shimmer_apq3  = np.std(rms) / (np.mean(rms) + 1e-8)

    harmonic, percussive = librosa.effects.hpss(y)
    hnr = 10 * np.log10((np.sum(harmonic**2) + 1e-8)
                        / (np.sum(percussive**2) + 1e-8))

    vq_feats = np.array([jitter_local, jitter_rap,
                         shimmer_local, shimmer_apq3, hnr])

    # ---- Energy block (7 features) ----
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    energy_feats = np.array([
        np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
        np.max(rms) - np.min(rms),
        np.mean(zcr), np.std(zcr),
    ])

    # ---- Spectral block (5 features) ----
    spectral_feats = np.array([
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_flatness(y=y)),
    ])

    # ---- Speech rate block (5 features) ----
    duration = len(y) / sr
    rms_full = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    peaks    = librosa.util.peak_pick(rms_full,
                                      pre_max=3, post_max=3,
                                      pre_avg=3, post_avg=5,
                                      delta=0.01, wait=5)
    syllable_count = len(peaks)
    speech_rate    = syllable_count / duration if duration > 0 else 0
    pause_ratio    = np.sum(rms_full < 0.02 * np.max(rms_full)) / max(len(rms_full), 1)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if np.isscalar(tempo) else float(np.atleast_1d(tempo)[0])
    except Exception:
        tempo = 0.0
    rate_feats = np.array([syllable_count, speech_rate, duration,
                           pause_ratio, tempo])

    # ---- Concatenate all blocks (52 + 6 + 5 + 7 + 5 + 5 = 80) ----
    return np.concatenate([mfcc_feats, pitch_feats, vq_feats,
                           energy_feats, spectral_feats, rate_feats])


FEATURE_NAMES = (
    [f"mfcc_mean_{i}"   for i in range(13)] +
    [f"mfcc_std_{i}"    for i in range(13)] +
    [f"mfcc_d1_{i}"     for i in range(13)] +
    [f"mfcc_d2_{i}"     for i in range(13)] +
    ["pitch_mean", "pitch_std", "pitch_min", "pitch_max",
     "pitch_range", "voiced_frac"] +
    ["jitter_local", "jitter_rap", "shimmer_local",
     "shimmer_apq3", "hnr"] +
    ["rms_mean", "rms_std", "rms_max", "rms_min", "rms_range",
     "zcr_mean", "zcr_std"] +
    ["spec_centroid", "spec_bandwidth", "spec_rolloff",
     "spec_contrast", "spec_flatness"] +
    ["syllable_count", "speech_rate", "duration",
     "pause_ratio", "tempo"]
)
assert len(FEATURE_NAMES) == 80


# ============================================================================
#  DATASET LOADER — RAVDESS
# ============================================================================
def load_ravdess(data_dir):
    """
    RAVDESS filename format: 03-01-06-01-02-01-12.wav
        [modality]-[channel]-[emotion]-[intensity]-[statement]-[rep]-[actor]
    We use: emotion (index 2) and actor (index 6).
    """
    records = []
    wavs = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    print(f"  Found {len(wavs)} .wav files in {data_dir}")

    for wav in wavs:
        name = os.path.basename(wav).replace(".wav", "")
        parts = name.split("-")
        if len(parts) < 7:
            continue
        emotion_code = parts[2]
        actor_id     = f"actor_{parts[6]}"
        if emotion_code not in RAVDESS_EMOTION_MAP:
            continue
        emotion_name, fatigue_label = RAVDESS_EMOTION_MAP[emotion_code]
        records.append({
            "path": wav, "speaker_id": actor_id,
            "emotion": emotion_name, "label": fatigue_label,
        })
    return pd.DataFrame(records)


# ============================================================================
#  STAGE 3 — BUILD DELTA FEATURES (THIS IS THE CRUX)
# ============================================================================
def compute_baselines(features_df):
    """
    Personal baseline = mean feature vector of that speaker's 'alert' clips.
    This represents that specific person's normal, rested voice.
    Returns: dict {speaker_id -> baseline_vector(80,)}
    """
    baselines = {}
    for spk, grp in features_df.groupby("speaker_id"):
        alert_clips = grp[grp["label"] == "alert"]
        if len(alert_clips) == 0:
            print(f"  ⚠ speaker {spk} has no 'alert' clips — skipped")
            continue
        feat_cols = [c for c in alert_clips.columns if c.startswith("f_")]
        baselines[spk] = alert_clips[feat_cols].mean().values
    return baselines


def build_delta_dataset(features_df, baselines):
    """
    For every clip: delta = clip_features - speaker's baseline
    This is what the classifier actually trains on.
    """
    feat_cols = [c for c in features_df.columns if c.startswith("f_")]
    X, y, groups = [], [], []

    for _, row in features_df.iterrows():
        spk = row["speaker_id"]
        if spk not in baselines:
            continue
        clip_feats = row[feat_cols].values.astype(float)
        delta      = clip_feats - baselines[spk]
        X.append(delta)
        y.append(LABEL_TO_INT[row["label"]])
        groups.append(spk)

    return np.array(X), np.array(y), np.array(groups)


# ============================================================================
#  PLOTTING HELPERS — one function per figure
# ============================================================================
def plot_label_distribution(meta_df, save_path):
    """Bar chart: how many clips per fatigue class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = meta_df["label"].value_counts().reindex(
        ["alert", "mild_fatigue", "fatigued"])
    colors = [LABEL_COLORS[c] for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution — Clips per Fatigue Label", fontsize=13)
    ax.set_ylabel("Number of clips")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_speaker_distribution(meta_df, save_path):
    """Stacked bar chart: clips per speaker, colored by class."""
    pivot = (meta_df.groupby(["speaker_id", "label"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["alert", "mild_fatigue", "fatigued"], fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[LABEL_COLORS[c] for c in pivot.columns],
               edgecolor="black", width=0.85)
    ax.set_title("Clips per Speaker (stacked by fatigue class)", fontsize=13)
    ax.set_ylabel("Number of clips")
    ax.set_xlabel("Speaker")
    ax.legend(title="Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_sample_waveform(meta_df, save_path, sr=SAMPLE_RATE):
    """Raw waveform + RMS envelope for one clip per class."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=False)
    for ax, label in zip(axes, ["alert", "mild_fatigue", "fatigued"]):
        subset = meta_df[meta_df["label"] == label]
        if len(subset) == 0:
            continue
        wav_path = subset.iloc[0]["path"]
        y, _ = librosa.load(wav_path, sr=sr, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        t = np.arange(len(y)) / sr
        ax.plot(t, y, color=LABEL_COLORS[label], linewidth=0.5, alpha=0.8)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_t = np.linspace(0, t[-1] if len(t) > 0 else 0, len(rms))
        ax.plot(rms_t, rms / (rms.max() + 1e-8), color="black",
                linewidth=1.5, label="energy envelope (norm.)")
        ax.set_title(f"Sample waveform — {label} "
                     f"({os.path.basename(wav_path)})", fontsize=11)
        ax.set_ylabel("amplitude")
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (seconds)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_feature_boxplots(features_df, save_path):
    """Boxplots of 6 interpretable features across the 3 classes."""
    key_features = ["f_pitch_mean", "f_pitch_std", "f_speech_rate",
                    "f_jitter_local", "f_shimmer_local", "f_rms_mean"]
    titles = ["Pitch mean (Hz)", "Pitch std (Hz)", "Speech rate (syl/s)",
              "Jitter local", "Shimmer local", "RMS energy (mean)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat, title in zip(axes.flatten(), key_features, titles):
        data = [features_df[features_df["label"] == lab][feat].values
                for lab in ["alert", "mild_fatigue", "fatigued"]]
        bp = ax.boxplot(data, labels=["alert", "mild", "fatigued"],
                        patch_artist=True, showfliers=False)
        for patch, lab in zip(bp["boxes"],
                               ["alert", "mild_fatigue", "fatigued"]):
            patch.set_facecolor(LABEL_COLORS[lab])
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Key Acoustic Features across Fatigue Classes",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_baseline_vs_fatigued(features_df, baselines, save_path):
    """For one speaker: baseline features vs. fatigued clip features (key subset)."""
    # Pick the speaker with the most fatigued clips
    best_spk = (features_df[features_df["label"] == "fatigued"]
                .groupby("speaker_id").size().idxmax())
    fat_clip = features_df[(features_df["speaker_id"] == best_spk) &
                           (features_df["label"] == "fatigued")].iloc[0]

    feat_cols = [c for c in features_df.columns if c.startswith("f_")]
    baseline_vec = baselines[best_spk]
    fatigued_vec = fat_clip[feat_cols].values.astype(float)

    # Pick interpretable subset
    show_idx = [FEATURE_NAMES.index(n) for n in
                ["pitch_mean", "pitch_std", "pitch_range",
                 "speech_rate", "jitter_local", "shimmer_local",
                 "rms_mean", "hnr", "pause_ratio", "tempo"]]
    show_names = [FEATURE_NAMES[i] for i in show_idx]

    x = np.arange(len(show_idx))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, [baseline_vec[i] for i in show_idx], width=0.4,
           label="Baseline (rested)", color=LABEL_COLORS["alert"],
           edgecolor="black")
    ax.bar(x + 0.2, [fatigued_vec[i] for i in show_idx], width=0.4,
           label="Fatigued clip", color=LABEL_COLORS["fatigued"],
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(show_names, rotation=30, ha="right")
    ax.set_title(f"Baseline vs. Fatigued Clip — {best_spk}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_delta_heatmap(X, y, save_path):
    """Heatmap: average delta vector for each class (shows what the model sees)."""
    class_avg = np.stack([X[y == c].mean(axis=0) for c in [0, 1, 2]])
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(class_avg, cmap="RdBu_r", center=0,
                xticklabels=FEATURE_NAMES,
                yticklabels=["alert", "mild_fatigue", "fatigued"],
                cbar_kws={"label": "avg Δ (clip − baseline)"}, ax=ax)
    ax.set_title("Average Delta Vector per Class — the signal the model learns",
                 fontsize=13)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_pca_2d(X, y, save_path):
    """2D PCA projection of the delta dataset — is the signal separable?"""
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_s)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, lab in INT_TO_LABEL.items():
        mask = y == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=lab, color=LABEL_COLORS[lab],
                   alpha=0.6, s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("2D PCA Projection of Delta Features", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_confusion_matrices(results, save_path):
    """Confusion matrices for all models in a grid."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["alert", "mild", "fatigued"],
                    yticklabels=["alert", "mild", "fatigued"],
                    cbar=False, ax=ax)
        ax.set_title(f"{name}\nF1={r['f1_macro']:.2f}, "
                     f"Recall(fat)={r['recall_fatigued']:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_model_comparison(results, save_path):
    """Bar chart comparing F1 macro and recall-fatigued across models."""
    names = list(results.keys())
    f1s = [results[n]["f1_macro"] for n in names]
    recs = [results[n]["recall_fatigued"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, f1s, width=0.4, label="F1 (macro)",
           color="#3498db", edgecolor="black")
    ax.bar(x + 0.2, recs, width=0.4, label="Recall (fatigued)",
           color="#e74c3c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Higher is better", fontsize=13)
    ax.legend()
    for i, (f, r) in enumerate(zip(f1s, recs)):
        ax.text(i - 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


def plot_feature_importance(model, model_name, save_path, top_k=20):
    """Top-K feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        print(f"    (skipped — {model_name} has no feature_importances_)")
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURE_NAMES[i] for i in idx][::-1],
            imp[idx][::-1], color="#9b59b6", edgecolor="black")
    ax.set_title(f"Top {top_k} Feature Importances — {model_name}",
                 fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"    ✓ saved {save_path}")


# ============================================================================
#  STAGE 4 & 5 — TRAIN + EVALUATE MULTIPLE MODELS
# ============================================================================
def train_and_evaluate(X, y, groups):
    """Train RF, XGB, SVM, MLP on delta features and return the best one."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale",
                   class_weight="balanced", probability=True,
                   random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64),
                             max_iter=500, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            n_jobs=-1, use_label_encoder=False)

    results = {}
    for name, model in models.items():
        print(f"\n---- Training {name} ----")
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        # KEY METRIC: recall on class 2 (fatigued)  → missing tired workers is dangerous
        recall_fatigued = recall_score(y_te, y_pred, labels=[2], average="macro")
        f1_macro        = f1_score(y_te, y_pred, average="macro")

        print(f"  F1 (macro)            : {f1_macro:.3f}")
        print(f"  Recall (fatigued class): {recall_fatigued:.3f}")
        print("  Classification report:")
        print(classification_report(y_te, y_pred,
              target_names=["alert", "mild_fatigue", "fatigued"], zero_division=0))

        results[name] = {
            "model": model, "f1_macro": f1_macro,
            "recall_fatigued": recall_fatigued,
            "y_true": y_te, "y_pred": y_pred,
        }

    # Pick best model: prioritize recall_fatigued, break ties with f1_macro
    best_name = max(results,
                    key=lambda k: (results[k]["recall_fatigued"],
                                   results[k]["f1_macro"]))
    print(f"\n✓ Best model: {best_name} "
          f"(recall_fatigued={results[best_name]['recall_fatigued']:.3f}, "
          f"f1_macro={results[best_name]['f1_macro']:.3f})")

    return results[best_name]["model"], scaler, best_name, results


# ============================================================================
#  STAGE 7 — INFERENCE (used by dashboard / optimizer)
# ============================================================================
def predict_fatigue(wav_path, speaker_id, model, scaler, baselines):
    """
    Given a new voice check-in, return the worker's fatigue level.
      1. Extract 80 features
      2. Subtract speaker's baseline  →  delta vector
      3. Scale and run through model  →  class probabilities
      4. Convert to a continuous fatigue score in [0, 1]
    """
    feats = extract_features(wav_path)
    if feats is None:
        raise ValueError("Audio too short or unreadable.")
    if speaker_id not in baselines:
        raise ValueError(f"No baseline found for speaker '{speaker_id}'. "
                         "Enroll them first with at least one rested clip.")

    delta    = (feats - baselines[speaker_id]).reshape(1, -1)
    delta_s  = scaler.transform(delta)
    proba    = model.predict_proba(delta_s)[0]   # [P_alert, P_mild, P_fatigued]

    # Continuous score ∈ [0, 1]:  weight by severity
    fatigue_score = float(proba[1] * 0.5 + proba[2] * 1.0)

    # Categorical label
    if fatigue_score < 0.35:  status = "alert"
    elif fatigue_score < 0.70: status = "mild_fatigue"
    else:                     status = "fatigued"

    return {
        "fatigue_score"  : round(fatigue_score, 3),
        "status"         : status,
        "probabilities"  : {
            "alert"       : round(float(proba[0]), 3),
            "mild_fatigue": round(float(proba[1]), 3),
            "fatigued"    : round(float(proba[2]), 3),
        },
    }


# ============================================================================
#  MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/RAVDESS",
                        help="Folder containing the RAVDESS dataset")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--cache", default="outputs/features_cache.csv",
                        help="Cache file so we don't re-extract features every run")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------- Stage 1: Load dataset metadata --------
    print("\n[Stage 1] Loading RAVDESS metadata...")
    meta_df = load_ravdess(args.data_dir)

    if len(meta_df) == 0:
        raise SystemExit(f"No audio files found in {args.data_dir}. "
                         "Download the dataset first (see top of file).")
    print(f"  Loaded {len(meta_df)} clips across "
          f"{meta_df['speaker_id'].nunique()} speakers.")
    print(f"  Label distribution:\n{meta_df['label'].value_counts().to_string()}")

    # Plot 1 & 2: dataset overview
    print("\n  Generating dataset overview plots...")
    plot_label_distribution(meta_df, plots_dir / "01_label_distribution.png")
    plot_speaker_distribution(meta_df, plots_dir / "02_speaker_distribution.png")
    plot_sample_waveform(meta_df, plots_dir / "03_sample_waveform.png")

    # -------- Stage 2: Extract features (with caching) --------
    print("\n[Stage 2] Extracting features (this takes a few minutes)...")
    if os.path.exists(args.cache):
        print(f"  Loading cached features from {args.cache}")
        features_df = pd.read_csv(args.cache)
    else:
        rows = []
        for i, r in meta_df.iterrows():
            if i % 50 == 0:
                print(f"    {i}/{len(meta_df)}")
            feats = extract_features(r["path"])
            if feats is None:
                continue
            row = {"path": r["path"], "speaker_id": r["speaker_id"],
                   "emotion": r["emotion"], "label": r["label"]}
            for name, val in zip(FEATURE_NAMES, feats):
                row[f"f_{name}"] = val
            rows.append(row)
        features_df = pd.DataFrame(rows)
        features_df.to_csv(args.cache, index=False)
        print(f"  Saved features to {args.cache}")
    print(f"  Feature matrix: {features_df.shape}")

    # Plot 4: feature boxplots across classes
    print("\n  Generating feature distribution plots...")
    plot_feature_boxplots(features_df, plots_dir / "04_feature_boxplots.png")

    # -------- Stage 3: Compute baselines + delta features --------
    print("\n[Stage 3] Computing personal baselines (rested voice per speaker)...")
    baselines = compute_baselines(features_df)
    print(f"  Built baselines for {len(baselines)} speakers.")

    # Plot 5: baseline vs fatigued for one speaker
    plot_baseline_vs_fatigued(features_df, baselines,
                               plots_dir / "05_baseline_vs_fatigued.png")

    print("\n[Stage 3b] Building delta feature dataset...")
    X, y, groups = build_delta_dataset(features_df, baselines)
    print(f"  X shape: {X.shape},  y distribution: {np.bincount(y)}")

    # Plots 6 & 7: delta dataset visualizations
    print("\n  Generating delta feature plots...")
    plot_delta_heatmap(X, y, plots_dir / "06_delta_feature_heatmap.png")
    plot_pca_2d(X, y, plots_dir / "07_pca_2d_deltas.png")

    # -------- Stages 4-5: Train + evaluate --------
    print("\n[Stage 4] Training models on DELTA features...")
    best_model, scaler, best_name, all_results = train_and_evaluate(X, y, groups)

    # Plots 8 & 9 & 10: model evaluation visualizations
    print("\n  Generating model evaluation plots...")
    plot_confusion_matrices(all_results, plots_dir / "08_confusion_matrices.png")
    plot_model_comparison(all_results, plots_dir / "09_model_comparison.png")
    plot_feature_importance(best_model, best_name,
                             plots_dir / "10_feature_importance.png")

    # -------- Stage 6: Save everything --------
    print("\n[Stage 6] Saving artifacts...")
    joblib.dump(best_model, os.path.join(args.out_dir, "fatigue_model.pkl"))
    joblib.dump(scaler,     os.path.join(args.out_dir, "scaler.pkl"))
    joblib.dump(baselines,  os.path.join(args.out_dir, "baselines.pkl"))
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump({"best_model": best_name,
                   "feature_names": FEATURE_NAMES,
                   "label_map": LABEL_TO_INT,
                   "n_speakers": len(baselines),
                   "n_samples": int(len(X))}, f, indent=2)
    print(f"  Saved model + scaler + baselines to {args.out_dir}/")
    print(f"  Saved 10 plots to {plots_dir}/")

    # -------- Stage 7: Demo inference on a held-out clip --------
    print("\n[Stage 7] Demo inference on one random clip:")
    demo_row = features_df.sample(1, random_state=1).iloc[0]
    result   = predict_fatigue(demo_row["path"], demo_row["speaker_id"],
                               best_model, scaler, baselines)
    print(f"  File      : {os.path.basename(demo_row['path'])}")
    print(f"  Speaker   : {demo_row['speaker_id']}")
    print(f"  True label: {demo_row['label']}")
    print(f"  Predicted : {result}")

    print("\n✓ Pipeline finished. You can now call predict_fatigue() on new .wav files.")
    print("  Output per check-in: fatigue_score ∈ [0,1], status label, and class probabilities.")


if __name__ == "__main__":
    main()
