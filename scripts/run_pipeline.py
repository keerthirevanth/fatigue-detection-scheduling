"""
End-to-end runner for the fatigue-detection ML pipeline (Phase 1).

    python scripts/run_pipeline.py --data_dir data/RAVDESS

Stages:
    1. load metadata            5. cross-validate models (speaker-independent)
    2. extract features (cached) 6. tune best model
    3. baselines + delta dataset 7. fit final model on all data + save artifacts
    4. exploratory plots         8. demo inference
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Windows consoles default to cp1252, which can't encode the ✓/±/Δ glyphs the
# pipeline prints. Force UTF-8 so logging never crashes the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Make `src` importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fatigue import (load_ravdess, extract_features, compute_baselines,
                         build_delta_dataset, cross_validate_models, select_best,
                         tune_model, fit_final, oof_report, predict_fatigue)
from src.fatigue.config import FEATURE_NAMES, FEATURE_PREFIX, LABEL_TO_INT
from src.fatigue import plots


def extract_all(meta_df, cache_path):
    """Extract 80 features per clip, with a CSV cache to avoid recompute."""
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    for i, r in meta_df.reset_index(drop=True).iterrows():
        if i % 50 == 0:
            print(f"    {i}/{len(meta_df)}")
        feats = extract_features(r["path"])
        if feats is None:
            continue
        row = {"path": r["path"], "speaker_id": r["speaker_id"],
               "emotion": r.get("emotion", ""), "label": r["label"]}
        row.update({f"{FEATURE_PREFIX}{n}": v for n, v in zip(FEATURE_NAMES, feats)})
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Saved features to {cache_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/RAVDESS")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--cache", default="outputs/features_cache.csv")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--tune", action="store_true",
                    help="run randomized hyper-parameter search on the best model")
    ap.add_argument("--show", action="store_true", help="also display plots")
    args = ap.parse_args()

    out = Path(args.out_dir); plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -- Stage 1: metadata --
    print("\n[1] Loading RAVDESS metadata...")
    meta = load_ravdess(args.data_dir)
    if len(meta) == 0:
        raise SystemExit(f"No audio in {args.data_dir}. Run scripts/download_data.py first.")
    print(f"  {len(meta)} clips / {meta['speaker_id'].nunique()} speakers")
    print(meta["label"].value_counts().to_string())
    plots.plot_label_distribution(meta, plots_dir / "01_label_distribution.png", args.show)
    plots.plot_speaker_distribution(meta, plots_dir / "02_speaker_distribution.png", args.show)

    # -- Stage 2: features --
    print("\n[2] Extracting features...")
    feats_df = extract_all(meta, args.cache)
    print(f"  feature matrix: {feats_df.shape}")
    plots.plot_feature_boxplots(feats_df, plots_dir / "03_feature_boxplots.png", args.show)

    # -- Stage 3: baselines + deltas (enrollment split → no leakage) --
    print("\n[3] Building enrollment baselines + delta dataset...")
    baselines, enroll_idx = compute_baselines(feats_df)
    X, y, groups, _ = build_delta_dataset(feats_df, baselines, exclude_index=enroll_idx)
    print(f"  baselines: {len(baselines)} speakers | "
          f"X: {X.shape} | class counts: {np.bincount(y)} | "
          f"({len(enroll_idx)} clips reserved for enrollment)")
    plots.plot_delta_heatmap(X, y, plots_dir / "04_delta_heatmap.png", args.show)
    plots.plot_pca_2d(X, y, plots_dir / "05_pca_2d_deltas.png", args.show)

    # -- Stage 4/5: speaker-independent cross-validation --
    print(f"\n[4] Cross-validating models ({args.n_splits}-fold StratifiedGroupKFold)...")
    results = cross_validate_models(X, y, groups, n_splits=args.n_splits)
    best_name = select_best(results)
    print(f"\n  Best (by recall on 'fatigued'): {best_name}")
    print(oof_report(results, best_name))
    plots.plot_confusion_matrices(results, plots_dir / "06_confusion_matrices.png", args.show)
    plots.plot_model_comparison(results, plots_dir / "07_model_comparison.png", args.show)

    # -- Stage 6: tune + fit final --
    print(f"\n[5] Fitting final model ({best_name})...")
    if args.tune:
        print("  Tuning with randomized search...")
        final_model = tune_model(best_name, X, y, groups, n_splits=args.n_splits)
    else:
        from src.fatigue.model import get_model_specs
        final_model = fit_final(get_model_specs()[best_name][0], X, y)
    plots.plot_feature_importance(final_model, best_name,
                                  plots_dir / "08_feature_importance.png", show=args.show)

    # -- Stage 7: save artifacts --
    print("\n[6] Saving artifacts...")
    joblib.dump(final_model, out / "fatigue_model.pkl")   # Pipeline (scaler+clf)
    joblib.dump(baselines,   out / "baselines.pkl")
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "cv": {k: {kk: v[kk] for kk in
                       ("f1_mean", "f1_std", "recall_fat_mean", "recall_fat_std")}
                   for k, v in results.items()},
            "feature_names": FEATURE_NAMES, "label_map": LABEL_TO_INT,
            "n_speakers": len(baselines), "n_samples": int(len(X)),
        }, f, indent=2)
    print(f"  → {out}/fatigue_model.pkl, baselines.pkl, metadata.json")

    # -- Stage 8: demo inference --
    print("\n[7] Demo inference:")
    demo = feats_df.sample(1, random_state=1).iloc[0]
    res = predict_fatigue(demo["path"], demo["speaker_id"], final_model, baselines)
    print(f"  file={os.path.basename(demo['path'])} speaker={demo['speaker_id']} "
          f"true={demo['label']}\n  pred={res}")
    print("\n✓ Pipeline finished.")


if __name__ == "__main__":
    main()
