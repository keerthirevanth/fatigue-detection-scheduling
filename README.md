# Voice-Based Worker Fatigue Detection → Fatigue-Aware Shift Scheduling

An end-to-end system that (1) detects a worker's fatigue level from a short voice
check-in and (2) uses that signal to build a fatigue-aware shift schedule, with a
natural-language agent on top. Built to span **ML/DL** (acoustic modelling),
**DS** (leakage-free evaluation), **Operations Research** (the scheduler), and
**GenAI/agents** (orchestration).

## Core idea — personal delta baselines

Raw acoustics vary hugely between people, so we normalise every clip against that
*same* speaker's rested voice:

```
delta = clip_features − speaker_baseline
```

The classifier learns *changes* ("pitch dropped, speech slowed, jitter rose"),
not absolute voice traits — removing the biggest noise source (inter-speaker
variability). Baselines come from a small **enrollment** set of rested clips,
mirroring real deployment.

## Architecture

```
SENSING (ML/DL)      voice → 80 acoustic features → delta → fatigue score/label
DECISION (OR)        Gurobi MILP shift scheduler, measured-fatigue overtime gate
AGENT (GenAI)        3 Claude tool-use agents: orchestrator + scheduling + monitoring
```

Measurement-driven, closed loop: workers on overtime are voice-checked every few
hours; the moment a check-in reports fatigue they stop and the scheduler re-solves
on who's still available. (An earlier *predicted* fatigue-state model was dropped
in favour of frequent real measurement.)

## Transfer-learning branch (`transfer-learning`)

An ablation comparing the 80 hand-crafted acoustic features against **wav2vec 2.0**
self-supervised embeddings (768-dim, mean-pooled) and the two combined — under the
**same** enrollment-baseline + speaker-grouped CV, so only the representation
changes. Rationale: with ~1k samples over 24 speakers, training a deep net from
scratch overfits; transfer learning reuses a representation pre-trained on ~960h of
speech and fits only a light head. The delta-baseline trick still applies (subtract
each speaker's *embedding* baseline).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-transfer.txt
python scripts/run_transfer_learning.py --data_dir data/RAVDESS
```

## Datasets

| Dataset | Role | Labels |
|---|---|---|
| **RAVDESS** (24 speakers) | proxy / pre-training | emotion → fatigue (documented proxy) |
| **Sleepy Language Corpus (SLC)** | real target | Karolinska Sleepiness Scale (KSS 1–9) |

RAVDESS is an *emotion* corpus used only as a stand-in until SLC access is
granted (see `docs/slc_data_request_email.md`). KSS gives genuine fatigue labels
and a continuous regression target.

## Layout

```
src/fatigue/      config, features, data, baselines, model, plots, infer
src/scheduler/    optimizer.py  (Gurobi MILP, measured-fatigue overtime gate)
src/agent/        state.py, tools.py, agents.py, orchestrator.py
scripts/          run_pipeline.py, demo_scheduler.py, demo_agents.py, download_data.py
tests/            test_features.py, test_optimizer.py, test_agent_tools.py
```

### Agent layer (GenAI)

Three Claude agents (Anthropic tool-use, `claude-opus-4-8`) over deterministic
tools: an **orchestrator** the supervisor talks to (owns human-approval commits),
a **scheduling agent** (runs Gurobi, diagnoses unmet demand), and a **monitoring
agent** (voice check-ins, borderline-reading judgment — conservative-only).
Safety rules are enforced in `src/agent/tools.py`, not in prompts: the fatigue
threshold pulls workers mechanically, agents can never supply a fatigue reading
(the sensor callback is the only source), never re-activate a fatigued worker,
and never commit a schedule without a named human approver.

```bash
python scripts/demo_agents.py     # needs ANTHROPIC_API_KEY (or `ant auth login`)
```

## Rigor (fixes over the first prototype)

- **No data leakage** — enrollment clips build the baseline and are never scored.
- **Speaker-independent CV** — `StratifiedGroupKFold`; no speaker in train *and* test.
- **Fold-safe scaling** — `StandardScaler` lives inside the CV Pipeline.
- **Honest metrics** — CV mean ± std, not one lucky split.
- **Tuning** — group-aware `RandomizedSearchCV` (`--tune`).
- Recall on **fatigued** is the primary metric (missing a tired worker is the costly error).

## Run

```bash
pip install -r requirements.txt
python scripts/download_data.py               # RAVDESS (~215 MB), once
python scripts/run_pipeline.py --data_dir data/RAVDESS --tune   # train + evaluate
python scripts/demo_scheduler.py              # reactive fatigue-aware scheduling demo
pytest -q                                     # fast sanity tests (no dataset needed)
```

Artifacts land in `outputs/` (`fatigue_model.pkl`, `baselines.pkl`,
`metadata.json`, and `plots/`).
