# Industry-impact simulation: fatigue-aware scheduling vs. traditional practice

**Question:** does fatigue-aware scheduling actually change operational outcomes,
compared to how shifts are typically staffed today?

## Method

Two scheduling policies run over the **same synthetic scenario** so they're
directly comparable:

- **Baseline ("traditional")** — fixed-rotation greedy assignment (least-worked
  person gets the next open slot), overtime allowed up to a flat legal cap
  (base 8h + up to 4h OT), **no fatigue awareness at all**.
- **Fatigue-aware (this project)** — the Gurobi MILP scheduler
  (`src/scheduler/optimizer.py`), with each worker's overtime eligibility gated
  by a same-day voice check-in reading.

Both policies run day-by-day over an 18-worker, 21-day scenario with randomized
demand spikes and sick-outs (`src/simulation/scenario.py`), repeated across
**10 random seeds** for statistical stability. A hidden **ground-truth fatigue
process** drives each worker silently in the background — never given to either
policy directly. It's used only to (a) generate the noisy voice check-in
reading the fatigue-aware policy receives, and (b) let the simulation judge,
after the fact, whether *either* policy — including the baseline, which never
measures fatigue — put an already-fatigued worker on overtime.

Reproduce: `python scripts/run_simulation.py --n_workers 18 --n_days 21 --seeds 10`

## Results (mean ± std across 10 seeds)

| metric | baseline | fatigue-aware | change |
|---|---:|---:|---:|
| unsafe overtime assignments | 118.9 ± 22.6 | 16.3 ± 5.8 | **−86.3%** |
| demand coverage rate | 0.88 ± 0.05 | 0.88 ± 0.05 | +0.05 pp (unchanged) |
| total overtime hours | 569.4 ± 84.6 | 180.8 ± 52.1 | −388.6 h (−68%) |
| unmet demand hours | 380.1 ± 147.2 | 378.0 ± 145.7 | ~unchanged |
| workload fairness (std-dev hours/worker) | 24.0 ± 4.5 | 12.0 ± 2.5 | −50% |

## Findings

1. **86% fewer unsafe (fatigued) overtime assignments** — the headline safety
   result. This isn't "zero," and that's honest: the check-in is noisy (a
   stand-in for real acoustic-model error), so a few readings still miss a
   truly fatigued worker. But it's an order-of-magnitude improvement over a
   policy that never checks at all.
2. **Coverage held steady while overtime dropped ~68%.** The fatigue-aware
   policy did *not* trade safety for throughput — it covered essentially the
   same demand with far less overtime. Some of this is the fatigue gate
   preventing wasteful "assign the tired person anyway" overtime; some of it is
   simply that a MILP finds a more efficient global assignment than a greedy
   rotation, independent of fatigue. Both effects are real and both matter, but
   they shouldn't be conflated — the safety result (86% fewer unsafe
   assignments) is the fatigue-specific claim; the overtime/fairness gains are
   partly an "optimization beats greedy heuristics" result.
3. **Workload got more even (fairness std-dev halved).** A side effect of
   MILP-based fair-load balancing that a greedy rotation doesn't achieve as well.
4. **Illustrative cost translation** (assumed $37.5/h overtime, $60/h unmet
   demand — labeled assumptions, not measured facts, and trivially overridable
   via `--ot_rate`/`--unmet_rate`): baseline ≈ $44,158 per 21-day horizon vs.
   fatigue-aware ≈ $29,461 — a **33% reduction**, driven almost entirely by the
   overtime-hours drop.

## Honest caveats

- This is a **synthetic simulation**, not a live deployment. The ground-truth
  fatigue process, demand pattern, and sick-out rate are modeled, not measured
  from a real plant.
- The **86% unsafe-assignment reduction and the coverage/overtime numbers are
  the credible, load-bearing results** — they follow directly from the
  fatigue-gating logic. The **dollar figures are illustrative**, built on
  assumed hourly rates; treat them as "here's how you'd translate this to cost
  once you plug in real rates," not as a validated savings claim.
- The overtime/fairness gains partly reflect MILP vs. greedy-heuristic quality,
  not fatigue-awareness alone — see finding 2.

## What this demonstrates

A system like this — voice-based fatigue sensing feeding a fatigue-gated
scheduler — can substantially reduce unsafe overtime assignments without
sacrificing demand coverage, evaluated against a realistic traditional-practice
baseline under randomized, repeated conditions.
