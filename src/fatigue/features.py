"""
Stage 1 & 2 — acoustic feature extraction.

Extracts an 80-dimensional feature vector from a single .wav file:
    52 MFCC (mean/std/delta/delta2)  +  6 pitch  +  5 voice-quality
    +  7 energy  +  5 spectral  +  5 speech-rate  =  80

The vector order is fixed by config.FEATURE_NAMES.
"""
import numpy as np
import librosa

from .config import SAMPLE_RATE, N_MFCC


def extract_features(wav_path, sr=SAMPLE_RATE):
    """Return an 80-dim feature vector for one clip, or None if the clip is too short."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # Clean: trim leading/trailing silence, then peak-normalize amplitude.
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr * 0.5:               # skip clips shorter than 0.5 s
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
