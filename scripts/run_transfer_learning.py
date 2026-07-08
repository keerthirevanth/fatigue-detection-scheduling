"""
Transfer-learning EXPERIMENT: compare acoustic representations for fatigue
detection under one leakage-free, speaker-independent protocol (enrollment
baselines + StratifiedGroupKFold). Nothing is assumed — every representation is
measured, and the best model per representation is additionally hyper-parameter
tuned so the winner is tuned, not just lucky.

Representations compared (identical clips / folds / enrollment — only the
feature vector changes):
    handcrafted_80        80 engineered acoustic features (the Phase-1 baseline)
    w2v_mean              wav2vec2 mean pooling                 (768)
    w2v_meanstd           wav2vec2 mean+std (statistics pool)   (1536)
    w2v_meanstdmax        wav2vec2 mean+std+max                 (2304)
    combined              handcrafted_80 + w2v_mean             (848)

    python scripts/run_transfer_learning.py --data_dir data/RAVDESS [--tune]

Needs torch+transformers. If torch was installed to a side directory to dodge
the Windows long-path limit, run with PYTHONPATH pointing at it, e.g.:
    PYTHONPATH=C:\\pt python scripts/run_transfer_learning.py --data_dir data/RAVDESS
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import gc

from sklearn.base import clone
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV

from src.fatigue import (load_ravdess, extract_features, compute_baselines,
                         build_delta_dataset, cross_validate_models, select_best)
from src.fatigue.model import get_model_specs, _pipe, FATIGUED
from src.fatigue.config import FEATURE_NAMES, FEATURE_PREFIX, RANDOM_STATE
from src.fatigue.embeddings import embedding_row, EMB_FEATURE_NAMES, MODEL_NAME, HIDDEN_DIM


def load_or_build_handcrafted(meta, cache):
    if Path(cache).exists():
        print(f"  hand-crafted: loading cache {cache}")
        return pd.read_csv(cache)
    print("  hand-crafted: extracting (no cache found)...")
    rows = []
    for i, r in meta.reset_index(drop=True).iterrows():
        if i % 100 == 0:
            print(f"    {i}/{len(meta)}")
        f = extract_features(r["path"])
        if f is None:
            continue
        row = {"path": r["path"], "speaker_id": r["speaker_id"],
               "emotion": r.get("emotion", ""), "label": r["label"]}
        row.update({f"{FEATURE_PREFIX}{n}": v for n, v in zip(FEATURE_NAMES, f)})
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load_or_build_embeddings(meta, cache):
    if Path(cache).exists():
        print(f"  embeddings: loading cache {cache}")
        return pd.read_csv(cache)
    print(f"  embeddings: extracting with {MODEL_NAME} (first run downloads ~360MB)...")
    rows = []
    for i, r in meta.reset_index(drop=True).iterrows():
        if i % 50 == 0:
            print(f"    {i}/{len(meta)}")
        row = embedding_row(r)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def make_delta(full, feat_cols):
    """Same enrollment/delta pipeline for any column subset of the aligned frame."""
    meta_cols = [c for c in ["path", "speaker_id", "emotion", "label"] if c in full.columns]
    sub = full[meta_cols + feat_cols]
    baselines, enroll = compute_baselines(sub)
    return build_delta_dataset(sub, baselines, exclude_index=enroll)


def tuned_scores(name, X, y, groups, n_splits, n_iter=20):
    """Group-aware randomized search with MULTI-METRIC scoring. Reads both F1 and
    recall(fatigued) — mean±std — straight from cv_results_ at the best index, so
    there's no redundant re-CV. `probability` is disabled during the search (we
    only need predict for scoring), which avoids SVC's costly internal Platt
    calibration on every fit."""
    est, dist = get_model_specs()[name]
    est = clone(est)
    if hasattr(est, "probability"):
        est.set_params(probability=False)
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": make_scorer(f1_score, average="macro"),
               "rec": make_scorer(recall_score, labels=[FATIGUED],
                                  average="macro", zero_division=0)}
    search = RandomizedSearchCV(_pipe(est), dist, n_iter=n_iter, scoring=scoring,
                                refit="f1", cv=cv, random_state=RANDOM_STATE, n_jobs=2)
    search.fit(X, y, groups=groups)
    i, r = search.best_index_, search.cv_results_
    return {"f1_mean": float(r["mean_test_f1"][i]), "f1_std": float(r["std_test_f1"][i]),
            "recall_fat_mean": float(r["mean_test_rec"][i]),
            "recall_fat_std": float(r["std_test_rec"][i]),
            "best_params": {k: v for k, v in search.best_params_.items()}}


def evaluate(full, feat_cols, name, n_splits, tune):
    X, y, groups, _ = make_delta(full, feat_cols)
    print(f"\n[{name}]  X={X.shape}  ({len(feat_cols)} features)")
    results = cross_validate_models(X, y, groups, n_splits=n_splits)
    best = select_best(results)
    row = {"name": name, "n_features": len(feat_cols), "best_model": best,
           "cv_untuned": {k: {kk: results[k][kk] for kk in
                              ("f1_mean", "f1_std", "recall_fat_mean", "recall_fat_std")}
                          for k in results}}
    if tune:
        print(f"  tuning {best} (group-aware randomized search, multi-metric)...")
        row["tuned_model"] = best
        row["cv_tuned"] = tuned_scores(best, X, y, groups, n_splits)
        print(f"    tuned F1={row['cv_tuned']['f1_mean']:.3f}±{row['cv_tuned']['f1_std']:.3f}"
              f"  recall(fat)={row['cv_tuned']['recall_fat_mean']:.3f}")
    del X, y, groups
    gc.collect()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/RAVDESS")
    ap.add_argument("--handcrafted_cache", default="outputs/features_cache.csv")
    ap.add_argument("--embeddings_cache", default="outputs/embeddings_cache.csv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--tune", action="store_true",
                    help="also hyper-parameter tune the best model per representation")
    args = ap.parse_args()

    print("[1] Loading metadata...")
    meta = load_ravdess(args.data_dir)
    if len(meta) == 0:
        raise SystemExit(f"No audio in {args.data_dir}.")
    print(f"  {len(meta)} clips / {meta['speaker_id'].nunique()} speakers")

    print("\n[2] Building both feature representations...")
    hand = load_or_build_handcrafted(meta, args.handcrafted_cache)
    emb = load_or_build_embeddings(meta, args.embeddings_cache)

    hand_cols = [c for c in hand.columns
                 if c.startswith(FEATURE_PREFIX) and "w2v" not in c]
    emb_cols_all = [f"{FEATURE_PREFIX}{n}" for n in EMB_FEATURE_NAMES]
    merged = hand.merge(emb[["path"] + emb_cols_all], on="path", how="inner") \
                 .reset_index(drop=True)
    print(f"  aligned clips present in both: {len(merged)}")

    # column subsets for each pooling view
    mean_cols = [f"{FEATURE_PREFIX}w2vmean_{i}" for i in range(HIDDEN_DIM)]
    std_cols = [f"{FEATURE_PREFIX}w2vstd_{i}" for i in range(HIDDEN_DIM)]
    max_cols = [f"{FEATURE_PREFIX}w2vmax_{i}" for i in range(HIDDEN_DIM)]

    representations = [
        ("handcrafted_80", hand_cols),
        ("w2v_mean", mean_cols),
        ("w2v_meanstd", mean_cols + std_cols),
        ("w2v_meanstdmax", mean_cols + std_cols + max_cols),
        ("combined", hand_cols + mean_cols),
    ]

    print(f"\n[3] Evaluating {len(representations)} representations "
          f"(tuning={'on' if args.tune else 'off'})...")
    runs = [evaluate(merged, cols, name, args.n_splits, args.tune)
            for name, cols in representations]

    # --- comparison table ---
    print("\n" + "=" * 84)
    header = f"{'representation':<18}{'#feat':>7}{'best model':>13}{'F1 (untuned)':>18}"
    if args.tune:
        header += f"{'F1 (tuned)':>18}"
    print(header)
    print("-" * 84)
    for r in runs:
        u = r["cv_untuned"][r["best_model"]]
        line = (f"{r['name']:<18}{r['n_features']:>7}{r['best_model']:>13}"
                f"{u['f1_mean']:>11.3f}±{u['f1_std']:<5.3f}")
        if args.tune and "cv_tuned" in r:
            t = r["cv_tuned"]
            line += f"{t['f1_mean']:>11.3f}±{t['f1_std']:<5.3f}"
        print(line)
    print("=" * 84)
    print("(primary safety metric is recall on 'fatigued' — see the JSON for full detail)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "transfer_comparison.json", "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\n✓ saved {out}/transfer_comparison.json")


if __name__ == "__main__":
    main()
