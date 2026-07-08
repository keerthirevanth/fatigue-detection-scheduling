"""
Transfer-learning feature extractor — self-supervised speech embeddings.

Instead of 80 hand-crafted acoustic features, run each clip through a frozen
wav2vec 2.0 encoder (pre-trained on ~960h of unlabeled speech) and mean-pool its
last hidden state into a 768-dim utterance embedding.

WHY THIS AND NOT A CNN FROM SCRATCH
-----------------------------------
We have ~1k samples from 24 speakers under speaker-independent CV — far too few
to train a deep net from scratch without memorising speakers. Transfer learning
sidesteps that: the representation was learned on huge unlabeled audio; we only
fit a light classifier head on our small labeled set. The delta-baseline trick
still applies — we subtract each speaker's *embedding* baseline, exactly as we
subtract their feature baseline.

The output frame uses the same `f_` column prefix as features.py, so the
enrollment-baseline + speaker-grouped CV code works on it unchanged.

Torch/transformers are imported lazily so importing this module never requires
them (only the extraction functions do).
"""
import numpy as np
import librosa

from .config import SAMPLE_RATE, FEATURE_PREFIX

MODEL_NAME = "facebook/wav2vec2-base"     # pre-trained SSL encoder (not ASR-tuned)
HIDDEN_DIM = 768                          # wav2vec2-base hidden size

# We store THREE pooling views of the encoder output per clip, so the experiment
# can compare pooling strategies (mean / mean+std / mean+std+max) by selecting
# columns — no re-extraction needed. Column groups are prefixed so the runner
# can slice them apart.
POOLS = ("mean", "std", "max")
EMB_FEATURE_NAMES = [f"w2v{pool}_{i}" for pool in POOLS for i in range(HIDDEN_DIM)]
EMB_DIM = len(EMB_FEATURE_NAMES)          # 2304

# Back-compat alias: the plain "mean" pool is the default single representation.
MEAN_FEATURE_NAMES = [f"w2vmean_{i}" for i in range(HIDDEN_DIM)]

_MODEL = None
_EXTRACTOR = None


def _load():
    """Lazily load the encoder + feature extractor once (cached in module globals)."""
    global _MODEL, _EXTRACTOR
    if _MODEL is None:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        _EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _MODEL = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        _MODEL.eval()
        torch.set_grad_enabled(False)
    return _EXTRACTOR, _MODEL


def extract_embedding(wav_path, sr=SAMPLE_RATE):
    """Return a 2304-dim vector = [mean(768) | std(768) | max(768)] of the
    wav2vec2 last hidden state over time, or None for too-short clips."""
    import torch

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr * 0.5:
        return None

    extractor, model = _load()
    inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state       # (1, T, 768)
    h = hidden.squeeze(0)                                 # (T, 768)
    pooled = torch.cat([h.mean(dim=0), h.std(dim=0), h.amax(dim=0)])  # (2304,)
    return pooled.cpu().numpy().astype(np.float32)


def embedding_row(meta_row):
    """Build one features_df-style row (path/speaker_id/emotion/label + f_w2v_i)
    for a metadata row, or None if the clip is too short."""
    emb = extract_embedding(meta_row["path"])
    if emb is None:
        return None
    row = {"path": meta_row["path"], "speaker_id": meta_row["speaker_id"],
           "emotion": meta_row.get("emotion", ""), "label": meta_row["label"]}
    row.update({f"{FEATURE_PREFIX}{n}": v for n, v in zip(EMB_FEATURE_NAMES, emb)})
    return row
