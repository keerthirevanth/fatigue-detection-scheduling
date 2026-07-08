"""
Fast, data-light sanity checks. Run with:  pytest -q

These don't need the full RAVDESS download — they synthesise a short tone so the
feature extractor and the delta/baseline plumbing can be tested in CI.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fatigue.config import FEATURE_NAMES, FEATURE_PREFIX
from src.fatigue.features import extract_features
from src.fatigue.baselines import compute_baselines, build_delta_dataset


def _write_tone(path, freq=150, sr=16000, secs=2.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, y, sr)


def test_feature_vector_length(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_tone(wav)
    feats = extract_features(str(wav))
    assert feats is not None
    assert feats.shape == (len(FEATURE_NAMES),) == (80,)
    assert np.isfinite(feats).all()


def test_short_clip_returns_none(tmp_path):
    wav = tmp_path / "short.wav"
    _write_tone(wav, secs=0.2)          # below the 0.5 s floor
    assert extract_features(str(wav)) is None


def _toy_features_df():
    """Two speakers, each with alert + fatigued clips (random feature values)."""
    rng = np.random.default_rng(0)
    rows = []
    for spk in ["actor_01", "actor_02"]:
        for label in ["alert", "alert", "fatigued", "fatigued"]:
            row = {"path": f"{spk}_{label}.wav", "speaker_id": spk, "label": label}
            row.update({f"{FEATURE_PREFIX}{n}": rng.normal() for n in FEATURE_NAMES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_enrollment_clips_excluded_from_scoring():
    df = _toy_features_df()
    baselines, enroll_idx = compute_baselines(df, enroll_frac=0.5)
    X, y, groups, paths = build_delta_dataset(df, baselines, exclude_index=enroll_idx)
    # every enrollment clip must be absent from the scored set
    assert len(enroll_idx) > 0
    assert X.shape[0] == len(df) - len(enroll_idx)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert set(np.unique(groups)).issubset({"actor_01", "actor_02"})
