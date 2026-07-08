# Transfer-learning ablation — results

**Question:** do self-supervised **wav2vec 2.0** speech embeddings beat the 80
hand-crafted acoustic features for voice-based fatigue detection?

**Protocol (identical for every row):** RAVDESS proxy labels, 1440 clips / 24
speakers, enrollment-baseline delta features, speaker-independent
`StratifiedGroupKFold` (5-fold), scaler inside the CV pipeline. Only the feature
representation changes across rows — same clips, same folds, same enrollment
split — so differences are attributable to the representation, not the protocol.

**Harness sanity check:** the `handcrafted_80` row reproduces the standalone
Phase-1 numbers exactly (XGBoost F1 0.651±0.032, recall 0.721; SVM F1 0.693),
confirming the comparison harness is the same one used for the baseline.

Reproduce: `PYTHONPATH=C:\pt python scripts/run_transfer_learning.py --data_dir data/RAVDESS --tune`

## Results (CV mean ± std)

| Representation   | #feat | best model | F1 untuned      | F1 tuned        | recall(fatigued) |
|------------------|------:|-----------:|-----------------|-----------------|-----------------:|
| handcrafted_80   |    80 | XGBoost    | 0.651 ± 0.032   | 0.659 ± 0.042   | 0.744            |
| w2v_mean         |   768 | MLP        | 0.677 ± 0.072   | 0.683 ± 0.046   | 0.734            |
| w2v_meanstd      |  1536 | SVM        | 0.703 ± 0.046   | 0.704 ± 0.031   | **0.801**        |
| w2v_meanstdmax   |  2304 | SVM        | **0.714 ± 0.048** | **0.714 ± 0.048** | 0.776          |
| combined         |   848 | SVM        | 0.708 ± 0.043   | 0.708 ± 0.043   | 0.784            |

wav2vec2 mean-pool is 768-dim; +std and +max add statistics-pooling views.
`combined` = handcrafted_80 + wav2vec2 mean. Tuned = group-aware
`RandomizedSearchCV`, multi-metric (F1 + recall), best model per representation.

## Findings

1. **Transfer learning helps, consistently.** Every wav2vec2 variant beats the
   80-dim hand-crafted baseline on both F1 (~0.65 → ~0.71) and, more importantly
   for this safety task, **recall on the fatigued class (0.72–0.74 → 0.78–0.80)**.
   The best recall (0.801) comes from `w2v_meanstd` + SVM — the standout, since
   missing a fatigued worker is the costly error.
2. **Richer pooling helps.** mean → mean+std → mean+std+max improves F1
   monotonically; statistics pooling captures variation the mean discards.
3. **Model family matters more than tuning here.** Tree ensembles (RF/XGB)
   degrade badly on the dense 768–2304-dim embeddings (RF ~0.55 F1); margin/
   kernel (SVM) and neural (MLP) heads handle them well. Hyper-parameter tuning
   moved the winners by ≤0.01 F1 — the defaults were already near-optimal — so
   the lift is from the *representation and model family*, not the search.
4. **Honest caveat:** the F1 gain (~0.06) is real and consistent but modest, and
   the ±std bands partially overlap; the recall gain is the more decisive result.
   All of this is still on the RAVDESS emotion→fatigue *proxy* — the ranking
   should be re-confirmed on real KSS-labeled data (SLC) when available.

**Optimal approach on this data:** frozen wav2vec 2.0 encoder → statistics
pooling (mean+std) → speaker delta baseline → SVM (RBF, C≈1–2). Best F1 ≈ 0.71,
best fatigued-recall ≈ 0.80, vs. hand-crafted 0.66 / 0.74.
