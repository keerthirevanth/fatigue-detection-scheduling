"""
Dataset loaders. Each loader returns a metadata DataFrame with a common schema:

    columns: path, speaker_id, label   (+ optional: emotion, kss)

- load_ravdess : emotion corpus used as a PROXY / pre-training set.
- load_slc     : Sleepy Language Corpus (KSS 1-9) — the REAL fatigue target.
                 Stubbed until data access is granted (see docs/slc_data_request_email.md).
"""
import os
import glob

import pandas as pd

from .config import RAVDESS_EMOTION_MAP


def load_ravdess(data_dir):
    """
    RAVDESS filename format: 03-01-06-01-02-01-12.wav
        [modality]-[channel]-[emotion]-[intensity]-[statement]-[rep]-[actor]
    We use emotion (index 2) and actor (index 6).
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


def load_slc(data_dir):
    """
    Sleepy Language Corpus loader — PLACEHOLDER.

    Once access is granted, the corpus ships with a labels file mapping each
    recording to a Karolinska Sleepiness Scale value (KSS, 1-9). The plan:

        kss 1-3  -> alert
        kss 4-6  -> mild_fatigue
        kss 7-9  -> fatigued
        (also keep raw kss as a regression target)

    Fill this in when the data arrives; the rest of the pipeline already accepts
    the (path, speaker_id, label[, kss]) schema.
    """
    raise NotImplementedError(
        "SLC loader not implemented yet — awaiting data access. "
        "See docs/slc_data_request_email.md."
    )
