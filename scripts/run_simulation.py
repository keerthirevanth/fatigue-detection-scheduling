"""
Industry-impact simulation: fatigue-aware scheduling (Gurobi + voice check-ins)
vs. a traditional fixed-rotation baseline, run over the SAME synthetic
multi-week scenario, averaged across multiple random seeds.

    python scripts/run_simulation.py --n_workers 18 --n_days 21 --seeds 10

Every number reported is measured from the simulation itself. The one place we
inject an assumption is translating hours into dollars (--ot_rate,
--unmet_rate) — those are labeled as assumptions, not measured facts, and are
easy to override.
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

from src.simulation.scenario import generate_scenario
from src.simulation.policies import baseline_policy, fatigue_aware_policy, summarize


def run_one_seed(seed, n_workers, n_days):
    scenario = generate_scenario(n_workers=n_workers, n_days=n_days, seed=seed)
    base = summarize(baseline_policy(scenario), scenario)
    aware = summarize(fatigue_aware_policy(scenario), scenario)
    base["seed"] = aware["seed"] = seed
    return base, aware


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_workers", type=int, default=18)
    ap.add_argument("--n_days", type=int, default=21)
    ap.add_argument("--seeds", type=int, default=10, help="number of random scenarios")
    ap.add_argument("--ot_rate", type=float, default=37.5, help="ASSUMED $/hr overtime cost")
    ap.add_argument("--unmet_rate", type=float, default=60.0,
                    help="ASSUMED $/hr cost of unmet demand (lost throughput)")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    print(f"Simulating {args.seeds} random {args.n_days}-day scenarios, "
          f"{args.n_workers} workers each (baseline vs. fatigue-aware)...\n")

    base_rows, aware_rows = [], []
    for seed in range(args.seeds):
        b, a = run_one_seed(seed, args.n_workers, args.n_days)
        base_rows.append(b); aware_rows.append(a)
        print(f"  seed {seed}: baseline unsafe={b['unsafe_overtime_assignments']:>3d}  "
              f"aware unsafe={a['unsafe_overtime_assignments']:>3d}  "
              f"baseline coverage={b['coverage_rate']:.3f}  "
              f"aware coverage={a['coverage_rate']:.3f}")

    base_df = pd.DataFrame(base_rows)
    aware_df = pd.DataFrame(aware_rows)

    def agg(df, col):
        return df[col].mean(), df[col].std()

    metrics = ["unsafe_overtime_assignments", "coverage_rate", "total_overtime_hours",
               "unmet_demand_hours", "fairness_stdev_hours"]
    print("\n" + "=" * 78)
    print(f"{'metric':<32}{'baseline (mean±std)':<24}{'fatigue-aware (mean±std)':<24}")
    print("-" * 78)
    comparison = {}
    for m in metrics:
        bm, bs = agg(base_df, m)
        am, as_ = agg(aware_df, m)
        comparison[m] = {"baseline_mean": round(bm, 3), "baseline_std": round(bs, 3),
                         "aware_mean": round(am, 3), "aware_std": round(as_, 3)}
        print(f"{m:<32}{f'{bm:.2f} ± {bs:.2f}':<24}{f'{am:.2f} ± {as_:.2f}':<24}")
    print("=" * 78)

    # ---- headline reductions ----
    unsafe_reduction = (1 - aware_df["unsafe_overtime_assignments"].mean()
                        / max(base_df["unsafe_overtime_assignments"].mean(), 1e-9)) * 100
    coverage_delta = (aware_df["coverage_rate"].mean() - base_df["coverage_rate"].mean()) * 100
    ot_delta = aware_df["total_overtime_hours"].mean() - base_df["total_overtime_hours"].mean()

    print(f"\nUnsafe (fatigued) overtime assignments: {unsafe_reduction:.1f}% lower "
          f"with fatigue-aware scheduling")
    print(f"Demand coverage change: {coverage_delta:+.2f} percentage points")
    print(f"Overtime hours change: {ot_delta:+.1f} h per {args.n_days}-day horizon "
          f"(avg over {args.seeds} seeds)")

    # ---- illustrative cost translation (ASSUMED rates, clearly labeled) ----
    base_cost = (base_df["total_overtime_hours"].mean() * args.ot_rate
                + base_df["unmet_demand_hours"].mean() * args.unmet_rate)
    aware_cost = (aware_df["total_overtime_hours"].mean() * args.ot_rate
                 + aware_df["unmet_demand_hours"].mean() * args.unmet_rate)
    print(f"\n[ILLUSTRATIVE — assumed ${args.ot_rate}/h overtime, ${args.unmet_rate}/h unmet demand]")
    print(f"  baseline cost/horizon:      ${base_cost:,.0f}")
    print(f"  fatigue-aware cost/horizon: ${aware_cost:,.0f}")
    print(f"  delta: ${aware_cost - base_cost:+,.0f} ({(aware_cost/base_cost - 1)*100:+.1f}%)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "comparison": comparison,
        "unsafe_reduction_pct": round(unsafe_reduction, 1),
        "coverage_delta_pp": round(coverage_delta, 2),
        "overtime_delta_hours": round(ot_delta, 1),
        "cost": {"baseline": round(base_cost, 0), "fatigue_aware": round(aware_cost, 0),
                "note": "illustrative — rates are assumptions, not measured"},
    }
    with open(out / "simulation_comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    base_df.to_csv(out / "simulation_baseline_seeds.csv", index=False)
    aware_df.to_csv(out / "simulation_aware_seeds.csv", index=False)
    print(f"\n✓ saved {out}/simulation_comparison.json (+ per-seed CSVs)")


if __name__ == "__main__":
    main()
