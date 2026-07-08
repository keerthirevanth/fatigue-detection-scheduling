"""
Plotting helpers — one function per figure. All figures are saved to disk;
interactive display is OFF by default (set show=True to also pop windows).

This module intentionally uses a non-interactive backend when imported so the
full pipeline can run headless (e.g. on a server) without blocking on plt.show().
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless-safe; never blocks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from .config import (FEATURE_NAMES, KEY_FEATURES, LABEL_COLORS, INT_TO_LABEL,
                     FEATURE_PREFIX, RANDOM_STATE)

sns.set_style("whitegrid")
_ORDER = ["alert", "mild_fatigue", "fatigued"]


def _finish(fig, save_path, show):
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"    ✓ saved {save_path}")


def plot_label_distribution(meta_df, save_path, show=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = meta_df["label"].value_counts().reindex(_ORDER).fillna(0)
    ax.bar(counts.index, counts.values,
           color=[LABEL_COLORS[c] for c in counts.index], edgecolor="black")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(int(v)), ha="center", fontweight="bold")
    ax.set_title("Class Distribution — Clips per Fatigue Label")
    ax.set_ylabel("Number of clips")
    _finish(fig, save_path, show)


def plot_speaker_distribution(meta_df, save_path, show=False):
    pivot = (meta_df.groupby(["speaker_id", "label"]).size()
             .unstack(fill_value=0)
             .reindex(columns=_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[LABEL_COLORS[c] for c in pivot.columns],
               edgecolor="black", width=0.85)
    ax.set_title("Clips per Speaker (stacked by fatigue class)")
    ax.set_ylabel("Number of clips"); ax.set_xlabel("Speaker")
    ax.legend(title="Label")
    plt.xticks(rotation=45, ha="right")
    _finish(fig, save_path, show)


def plot_feature_boxplots(features_df, save_path, show=False):
    feats = [FEATURE_PREFIX + n for n in
             ["pitch_mean", "pitch_std", "speech_rate",
              "jitter_local", "shimmer_local", "rms_mean"]]
    titles = ["Pitch mean (Hz)", "Pitch std (Hz)", "Speech rate (syl/s)",
              "Jitter local", "Shimmer local", "RMS energy (mean)"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat, title in zip(axes.flatten(), feats, titles):
        data = [features_df[features_df["label"] == lab][feat].values for lab in _ORDER]
        bp = ax.boxplot(data, labels=["alert", "mild", "fatigued"],
                        patch_artist=True, showfliers=False)
        for patch, lab in zip(bp["boxes"], _ORDER):
            patch.set_facecolor(LABEL_COLORS[lab]); patch.set_alpha(0.7)
        ax.set_title(title)
    fig.suptitle("Key Acoustic Features across Fatigue Classes", y=1.00)
    _finish(fig, save_path, show)


def plot_delta_heatmap(X, y, save_path, show=False):
    class_avg = np.stack([X[y == c].mean(axis=0) for c in [0, 1, 2]])
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(class_avg, cmap="RdBu_r", center=0, xticklabels=FEATURE_NAMES,
                yticklabels=_ORDER, cbar_kws={"label": "avg Δ (clip − baseline)"}, ax=ax)
    ax.set_title("Average Delta Vector per Class — the signal the model learns")
    plt.xticks(rotation=90, fontsize=6)
    _finish(fig, save_path, show)


def plot_pca_2d(X, y, save_path, show=False):
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_s)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, lab in INT_TO_LABEL.items():
        m = y == c
        ax.scatter(X_2d[m, 0], X_2d[m, 1], label=lab, color=LABEL_COLORS[lab],
                   alpha=0.6, s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("2D PCA Projection of Delta Features"); ax.legend()
    _finish(fig, save_path, show)


def plot_confusion_matrices(results, save_path, show=False):
    """results: name -> {y_true, y_pred, f1_mean, recall_fat_mean} (out-of-fold)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["alert", "mild", "fatigued"],
                    yticklabels=["alert", "mild", "fatigued"], cbar=False, ax=ax)
        ax.set_title(f"{name}\nF1={r['f1_mean']:.2f}, "
                     f"Recall(fat)={r['recall_fat_mean']:.2f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    _finish(fig, save_path, show)


def plot_model_comparison(results, save_path, show=False):
    names = list(results.keys())
    f1s = [results[n]["f1_mean"] for n in names]
    recs = [results[n]["recall_fat_mean"] for n in names]
    f1e = [results[n]["f1_std"] for n in names]
    rece = [results[n]["recall_fat_std"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, f1s, 0.4, yerr=f1e, capsize=4, label="F1 (macro)",
           color="#3498db", edgecolor="black")
    ax.bar(x + 0.2, recs, 0.4, yerr=rece, capsize=4, label="Recall (fatigued)",
           color="#e74c3c", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (CV mean ± std)")
    ax.set_title("Model Comparison — speaker-independent CV")
    ax.legend()
    _finish(fig, save_path, show)


def plot_feature_importance(model, model_name, save_path, top_k=20, show=False):
    """model may be a Pipeline; we look at its final 'clf' step."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        print(f"    (skipped importance — {model_name} has none)")
        return
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURE_NAMES[i] for i in idx][::-1], imp[idx][::-1],
            color="#9b59b6", edgecolor="black")
    ax.set_title(f"Top {top_k} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    _finish(fig, save_path, show)
