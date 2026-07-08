# Transfer-learning ablation — results

**Question:** do pretrained speech encoders beat the 80 hand-crafted acoustic
features for voice-based fatigue detection, and which backbone is best?

**Backbones compared:** wav2vec 2.0 (`facebook/wav2vec2-base`, SSL), HuBERT
(`facebook/hubert-base-ls960`, SSL), Whisper encoder (`openai/whisper-base`,
ASR). Each clip → frozen encoder → statistics pooling (mean / +std / +max) over
time → speaker delta baseline. Whisper pads to 30 s, so only the valid
(non-padding) frames are pooled.

**Protocol (identical for every row):** RAVDESS proxy labels, 1440 clips / 24
speakers, enrollment-baseline delta features, speaker-independent
`StratifiedGroupKFold` (5-fold), scaler inside the CV pipeline, best model per
representation additionally tuned via group-aware multi-metric
`RandomizedSearchCV`. Only the representation changes across rows — same clips,
folds, enrollment split — so differences are attributable to the representation.

**Harness sanity check:** the `handcrafted_80` row reproduces the standalone
Phase-1 numbers exactly (XGBoost F1 0.651±0.032, recall 0.721; SVM F1 0.693),
confirming the comparison harness is the same one used for the baseline.

Reproduce (torch on this machine lives in `C:\pt` — see note at bottom):

    PYTHONPATH=C:\pt python scripts/run_transfer_learning.py --data_dir data/RAVDESS --backbone hubert --tune

## Results — best model per representation (CV mean ± std, tuned)

| Backbone | representation      | #feat | model  | F1 (tuned)      | recall(fatigued) |
|----------|---------------------|------:|--------|-----------------|-----------------:|
| —        | handcrafted_80      |    80 | XGBoost| 0.659 ± 0.042   | 0.744            |
| wav2vec2 | w2v_mean            |   768 | MLP    | 0.683 ± 0.046   | 0.734            |
| wav2vec2 | w2v_meanstd         |  1536 | SVM    | 0.704 ± 0.031   | 0.801            |
| wav2vec2 | w2v_meanstdmax      |  2304 | SVM    | 0.714 ± 0.048   | 0.776            |
| wav2vec2 | combined            |   848 | SVM    | 0.708 ± 0.043   | 0.784            |
| **hubert**   | hubert_mean         |   768 | SVM | 0.788 ± 0.042   | 0.818            |
| **hubert**   | hubert_meanstd      |  1536 | SVM | 0.793 ± 0.037   | 0.815            |
| **hubert**   | hubert_meanstdmax   |  2304 | SVM | **0.794 ± 0.038** | 0.818          |
| **hubert**   | combined_hubert     |   848 | SVM | 0.793 ± 0.042   | **0.819**        |
| whisper  | whisper_mean        |   512 | SVM    | 0.793 ± 0.033   | 0.798            |
| whisper  | whisper_meanstd     |  1024 | XGBoost| 0.768 ± 0.036   | 0.781            |
| whisper  | whisper_meanstdmax  |  1536 | SVM    | 0.789 ± 0.011   | 0.814            |
| whisper  | combined_whisper    |   592 | SVM    | **0.800 ± 0.034** | 0.796          |

## Findings

1. **Ranking: HuBERT ≈ Whisper ≫ wav2vec2 > hand-crafted.** Both HuBERT and
   Whisper jump to ~0.79–0.80 F1 and ~0.82 recall on the fatigued class — a large,
   consistent gain over wav2vec2 (0.71) and hand-crafted (0.66). Since missing a
   fatigued worker is the costly error, the recall lift (0.74 → 0.82) matters most.
2. **HuBERT is the most robust backbone.** Every HuBERT pooling variant lands
   ~0.79 F1 / ~0.82 recall — the choice of pooling barely matters. This is the
   safest pick and matches the literature (HuBERT's masked-prediction pretraining
   captures paralinguistic/state cues well).
3. **Whisper is competitive but less stable.** Its best single score is the
   highest F1 (combined 0.800), but `whisper_meanstd` dips to 0.768 — its
   ASR-oriented features don't gain as cleanly from statistics pooling, and mean
   pooling alone is already near its best.
4. **Model family still matters.** SVM (RBF) wins almost everywhere on the dense
   embeddings; tree ensembles lag. Tuning moved winners by ≤0.02 F1 — the lift is
   the *representation*, not the hyper-parameter search.
5. **Honest caveats:** all results are on the RAVDESS emotion→fatigue *proxy*;
   ±std bands for HuBERT vs Whisper overlap (they are statistically indistinct
   here). The ranking should be re-confirmed on real KSS-labeled data (SLC).

**Recommended approach on this data:** frozen **HuBERT** encoder → mean (or
mean+std) pooling → speaker delta baseline → SVM (RBF). F1 ≈ 0.79, fatigued-recall
≈ 0.82, vs. hand-crafted 0.66 / 0.74 and wav2vec2 0.71 / 0.78.
