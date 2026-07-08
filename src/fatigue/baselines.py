"""
Stage 3 — personal baselines and DELTA features (the crux of the method).

Idea: raw acoustics vary hugely between people, so we normalise each clip
against that *same* speaker's rested ("alert") voice:

        delta = clip_features - speaker_baseline

LEAKAGE FIX vs. the original prototype
--------------------------------------
The old code computed each speaker's baseline from *all* their alert clips and
then also scored those same clips — so 'alert' deltas were ~0 by construction
(trivially separable) and the baseline "saw" the test clips.

Here we split each speaker's alert clips into:
    * an ENROLLMENT subset  -> used ONLY to build the baseline (never scored)
    * the remaining clips   -> scored like any other clip
This mirrors real deployment: a new worker records a few rested clips at
enrollment, and every later check-in is compared against that baseline.
"""
import numpy as np

from .config import LABEL_TO_INT, FEATURE_PREFIX, RANDOM_STATE


def _feature_columns(df):
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def compute_baselines(features_df, enroll_frac=0.5, random_state=RANDOM_STATE,
                      alert_label="alert"):
    """
    Build a per-speaker baseline vector from a subset of that speaker's alert clips.

    Returns
    -------
    baselines    : dict {speaker_id -> baseline_vector (n_features,)}
    enroll_index : set of DataFrame indices used for enrollment (to exclude
                   from scoring so we never evaluate on the baseline's own clips)
    """
    rng = np.random.default_rng(random_state)
    feat_cols = _feature_columns(features_df)
    baselines, enroll_index = {}, set()

    for spk, grp in features_df.groupby("speaker_id"):
        alert_clips = grp[grp["label"] == alert_label]
        if len(alert_clips) == 0:
            print(f"  ⚠ speaker {spk} has no '{alert_label}' clips — skipped")
            continue

        # Reserve ~enroll_frac of the alert clips for the baseline (at least 1).
        n_enroll = max(1, int(round(len(alert_clips) * enroll_frac)))
        enroll_rows = rng.choice(alert_clips.index.values, size=n_enroll, replace=False)

        baselines[spk] = alert_clips.loc[enroll_rows, feat_cols].mean().values
        enroll_index.update(enroll_rows.tolist())

    return baselines, enroll_index


def build_delta_dataset(features_df, baselines, exclude_index=None):
    """
    For every clip NOT reserved for enrollment: delta = clip_features - baseline.

    Returns
    -------
    X       : (n_samples, n_features) delta matrix
    y       : (n_samples,) integer labels
    groups  : (n_samples,) speaker ids  — pass to GroupKFold to keep a speaker
              entirely within one fold (speaker-independent evaluation)
    paths   : (n_samples,) source file paths (handy for error analysis)
    """
    exclude_index = exclude_index or set()
    feat_cols = _feature_columns(features_df)
    X, y, groups, paths = [], [], [], []

    for idx, row in features_df.iterrows():
        if idx in exclude_index:
            continue
        spk = row["speaker_id"]
        if spk not in baselines:
            continue
        clip_feats = row[feat_cols].values.astype(float)
        X.append(clip_feats - baselines[spk])
        y.append(LABEL_TO_INT[row["label"]])
        groups.append(spk)
        paths.append(row.get("path", ""))

    return np.array(X), np.array(y), np.array(groups), np.array(paths)
