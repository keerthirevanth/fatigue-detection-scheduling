"""
Transfer-learning feature extractor — self-supervised / pretrained speech
encoders, swappable by backbone.

For each clip we run a frozen encoder and pool its hidden states over time into
[mean | std | max] statistics-pooling views. The output frame uses the same
`f_` column prefix as features.py, so the enrollment-baseline + speaker-grouped
CV code works on it unchanged. The delta-baseline trick still applies — we
subtract each speaker's *embedding* baseline.

BACKBONES
---------
  wav2vec2  facebook/wav2vec2-base       768-dim, SSL (not ASR-tuned)
  hubert    facebook/hubert-base-ls960   768-dim, SSL (masked-prediction)
  whisper   openai/whisper-base          512-dim, ASR encoder (log-mel input)

wav2vec2/hubert take raw waveform; whisper takes log-mel and pads to 30s, so we
pool only the valid (non-padding) frames to avoid diluting short clips with
silence.

Torch/transformers are imported lazily so importing this module never needs them.
"""
import numpy as np
import librosa

from .config import SAMPLE_RATE, FEATURE_PREFIX

# name -> (hf id, kind, short tag, hidden dim)
BACKBONES = {
    "wav2vec2": ("facebook/wav2vec2-base",     "waveform", "w2v",  768),
    "hubert":   ("facebook/hubert-base-ls960", "waveform", "hub",  768),
    "whisper":  ("openai/whisper-base",        "whisper",  "whis", 512),
}
POOLS = ("mean", "std", "max")
DEFAULT_BACKBONE = "wav2vec2"

_LOADED = {}   # backbone -> (feature_extractor, model, kind)


def feature_names(backbone=DEFAULT_BACKBONE):
    _, _, tag, hdim = BACKBONES[backbone]
    return [f"{tag}{pool}_{i}" for pool in POOLS for i in range(hdim)]


def emb_dim(backbone=DEFAULT_BACKBONE):
    return len(POOLS) * BACKBONES[backbone][3]


# --- back-compat constants (default backbone = wav2vec2) ------------------
MODEL_NAME = BACKBONES[DEFAULT_BACKBONE][0]
HIDDEN_DIM = BACKBONES[DEFAULT_BACKBONE][3]
EMB_FEATURE_NAMES = feature_names(DEFAULT_BACKBONE)
EMB_DIM = emb_dim(DEFAULT_BACKBONE)
MEAN_FEATURE_NAMES = [f"{BACKBONES[DEFAULT_BACKBONE][2]}mean_{i}" for i in range(HIDDEN_DIM)]


def _load(backbone):
    """Lazily load a backbone's encoder + feature extractor once."""
    if backbone not in _LOADED:
        import torch
        from transformers import AutoModel, AutoFeatureExtractor
        hf_id, kind, _, _ = BACKBONES[backbone]
        fe = AutoFeatureExtractor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id)
        if kind == "whisper":
            model = model.get_encoder()          # encoder only
        model.eval()
        torch.set_grad_enabled(False)
        _LOADED[backbone] = (fe, model, kind)
    return _LOADED[backbone]


def _pool(hidden):
    """hidden: torch (T, H) -> numpy [mean|std|max] over time (3H,)."""
    import torch
    pooled = torch.cat([hidden.mean(dim=0), hidden.std(dim=0), hidden.amax(dim=0)])
    return pooled.cpu().numpy().astype(np.float32)


def extract_embedding(wav_path, backbone=DEFAULT_BACKBONE, sr=SAMPLE_RATE):
    """Return a (3*hidden_dim,) statistics-pooled embedding, or None if too short."""
    import torch

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr * 0.5:
        return None

    fe, model, kind = _load(backbone)
    with torch.no_grad():
        if kind == "whisper":
            feats = fe(y, sampling_rate=sr, return_tensors="pt")
            enc = model(feats.input_features).last_hidden_state[0]   # (1500, H)
            # whisper pads to 30s; keep only frames covering real audio (~320 samples/frame)
            n_valid = int(min(enc.shape[0], max(1, round(len(y) / 320))))
            hidden = enc[:n_valid]
        else:                                                        # wav2vec2 / hubert
            inputs = fe(y, sampling_rate=sr, return_tensors="pt")
            hidden = model(**inputs).last_hidden_state[0]            # (T, H)
    return _pool(hidden)


def embedding_row(meta_row, backbone=DEFAULT_BACKBONE):
    """Build one features_df-style row (path/speaker_id/emotion/label + f_ cols)
    for a metadata row, or None if the clip is too short."""
    emb = extract_embedding(meta_row["path"], backbone=backbone)
    if emb is None:
        return None
    row = {"path": meta_row["path"], "speaker_id": meta_row["speaker_id"],
           "emotion": meta_row.get("emotion", ""), "label": meta_row["label"]}
    row.update({f"{FEATURE_PREFIX}{n}": v
                for n, v in zip(feature_names(backbone), emb)})
    return row
