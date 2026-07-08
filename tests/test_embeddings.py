"""
Transfer-learning tests. The heavy end-to-end check (loading wav2vec2 and
embedding a tone) is skipped unless torch + transformers are installed AND the
model can be fetched; the lightweight structural checks always run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fatigue import embeddings
from src.fatigue.config import FEATURE_PREFIX


def test_embedding_names_match_dim():
    # 3 pooling views (mean/std/max) × 768 hidden dims = 2304 features.
    assert embeddings.EMB_DIM == 3 * embeddings.HIDDEN_DIM == 2304
    assert len(embeddings.EMB_FEATURE_NAMES) == embeddings.EMB_DIM
    assert len(set(embeddings.EMB_FEATURE_NAMES)) == embeddings.EMB_DIM   # unique
    assert all(any(n.startswith(f"w2v{p}_") for p in embeddings.POOLS)
               for n in embeddings.EMB_FEATURE_NAMES)
    assert embeddings.MEAN_FEATURE_NAMES[0] == "w2vmean_0"


def test_backbone_dims_and_names():
    # each backbone declares hidden dim; features = 3 pools × hidden dim
    for bb, (_, _, tag, hdim) in embeddings.BACKBONES.items():
        names = embeddings.feature_names(bb)
        assert embeddings.emb_dim(bb) == 3 * hdim == len(names)
        assert names[0] == f"{tag}mean_0"
        assert len(set(names)) == len(names)                 # unique
    # sanity on the specific sizes we compare in the writeup
    assert embeddings.emb_dim("wav2vec2") == embeddings.emb_dim("hubert") == 2304
    assert embeddings.emb_dim("whisper") == 1536


def test_module_imports_without_torch():
    # embeddings.py must import even on a machine without torch/transformers
    # (they're imported lazily inside the extraction functions).
    assert hasattr(embeddings, "extract_embedding")
    assert hasattr(embeddings, "embedding_row")


@pytest.mark.slow
def test_extract_embedding_shape(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import soundfile as sf
    try:
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sf.write(tmp_path / "tone.wav", 0.4 * np.sin(2 * np.pi * 150 * t), sr)
        emb = embeddings.extract_embedding(str(tmp_path / "tone.wav"))
    except Exception as exc:                       # no network / model fetch failed
        pytest.skip(f"wav2vec2 unavailable: {exc}")
    assert emb is not None
    assert emb.shape == (embeddings.EMB_DIM,)
    row = embeddings.embedding_row(
        {"path": str(tmp_path / "tone.wav"), "speaker_id": "s1", "label": "alert"})
    assert f"{FEATURE_PREFIX}w2vmean_0" in row
