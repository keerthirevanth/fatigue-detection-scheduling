"""
Central configuration: constants, label maps, and the canonical feature-name list.

Keeping every "magic" value in one place means the feature extractor, the model
trainer, and the inference function can never disagree about (for example) how
many features there are or what integer a label maps to.
"""

# ----------------------------------------------------------------- audio / DSP
SAMPLE_RATE = 16_000        # all audio resampled to this rate before feature extraction
N_MFCC = 13                 # number of MFCC coefficients (drives 52 of the 80 features)
RANDOM_STATE = 42           # single source of randomness for reproducibility

# ----------------------------------------------------------------- labels
# NOTE: RAVDESS is an *emotion* corpus. We use it only as a proxy / pre-training
# set for the acoustic model. The real fatigue labels come from the Sleepy
# Language Corpus (KSS 1-9). This mapping is the documented proxy, not ground truth.
RAVDESS_EMOTION_MAP = {
    "01": ("neutral",   "alert"),          # neutral   -> ALERT
    "02": ("calm",      "alert"),          # calm      -> ALERT (natural baseline)
    "03": ("happy",     "alert"),          # happy     -> ALERT
    "04": ("sad",       "fatigued"),       # sad       -> FATIGUED (low energy, flat)
    "05": ("angry",     "alert"),          # angry     -> ALERT (high arousal)
    "06": ("fearful",   "mild_fatigue"),   # fearful   -> MILD (unstable voice)
    "07": ("disgust",   "fatigued"),       # disgust   -> FATIGUED (low energy)
    "08": ("surprised", "alert"),          # surprised -> ALERT
}

LABEL_TO_INT = {"alert": 0, "mild_fatigue": 1, "fatigued": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
LABEL_COLORS = {"alert": "#2ecc71", "mild_fatigue": "#f39c12", "fatigued": "#e74c3c"}

# Prefix used for feature columns in the cached features DataFrame.
FEATURE_PREFIX = "f_"

# ----------------------------------------------------------------- feature names
# The order here MUST match the concatenation order in features.extract_features.
FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(N_MFCC)] +
    [f"mfcc_std_{i}"  for i in range(N_MFCC)] +
    [f"mfcc_d1_{i}"   for i in range(N_MFCC)] +
    [f"mfcc_d2_{i}"   for i in range(N_MFCC)] +
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
assert len(FEATURE_NAMES) == 80, f"expected 80 features, got {len(FEATURE_NAMES)}"

# Interpretable subset used for boxplots / baseline-vs-fatigued comparison plots.
KEY_FEATURES = [
    "pitch_mean", "pitch_std", "pitch_range", "speech_rate",
    "jitter_local", "shimmer_local", "rms_mean", "hnr", "pause_ratio", "tempo",
]

# Feature blocks — used by the feature-block ablation. (name -> list of feature names)
FEATURE_BLOCKS = {
    "mfcc":          [n for n in FEATURE_NAMES if n.startswith("mfcc_")],
    "pitch":         ["pitch_mean", "pitch_std", "pitch_min", "pitch_max",
                      "pitch_range", "voiced_frac"],
    "voice_quality": ["jitter_local", "jitter_rap", "shimmer_local",
                      "shimmer_apq3", "hnr"],
    "energy":        ["rms_mean", "rms_std", "rms_max", "rms_min", "rms_range",
                      "zcr_mean", "zcr_std"],
    "spectral":      ["spec_centroid", "spec_bandwidth", "spec_rolloff",
                      "spec_contrast", "spec_flatness"],
    "speech_rate":   ["syllable_count", "speech_rate", "duration",
                      "pause_ratio", "tempo"],
}
