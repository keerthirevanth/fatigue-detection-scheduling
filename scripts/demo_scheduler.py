"""
Demo for the measurement-driven fatigue-aware scheduler + the reactive OVERTIME
check-in loop.

Scenario: a plant runs 3×8h shifts over 3 departments, but is short-staffed, so
demand can only be met if some willing workers take overtime. While on overtime,
each worker is voice-checked every few hours. When a check-in reports fatigue, the
worker stops (becomes unavailable) and we RE-SOLVE on who's left — showing how the
schedule reacts to fatigue measured in real time.

In production the check-in calls src.fatigue.predict_fatigue on the live voice
sample. Here we MOCK that reading: fatigue rises with cumulative overtime hours.

    python scripts/demo_scheduler.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src.scheduler.optimizer import (Worker, Shift, SchedulingProblem, SchedParams,
                                      build_and_solve, ext_cap)

DEPARTMENTS = ["assembly", "packaging", "quality"]
SLOTS = [("morning", 6.0), ("evening", 14.0), ("night", 22.0)]


def make_shifts(day=1):
    return [Shift(f"d{day}_{slot}", day, slot, clk, 8.0, slot == "night")
            for slot, clk in SLOTS]


def base_demand(day=1):
    d = {}
    for slot, _ in SLOTS:
        sid = f"d{day}_{slot}"
        d[(sid, "assembly")]  = 16.0
        d[(sid, "packaging")] = 8.0
        d[(sid, "quality")]   = 8.0
    return d


def show(schedule, summary, title):
    print(f"\n===== {title} =====")
    if schedule.empty:
        print("  (no assignments)")
    else:
        for _, r in schedule.iterrows():
            tag = f"  +{r['overtime_hours']}h OT" if r["overtime"] else ""
            print(f"  {r['slot']:8s} | {r['department']:10s} | {r['worker']} "
                  f"({r['total_hours']}h){tag}")
    print(f"  unmet: {summary['total_unmet_hours']}h | "
          f"overtime: {summary['total_overtime_hours']}h | "
          f"workers used: {summary['workers_used']} | obj: {summary['objective']}")


# Per-worker fatigue build-up rate (per overtime hour) — some people tire faster.
OT_FATIGUE_RATE = {"W01": 0.10, "W02": 0.16, "W03": 0.12}


def mock_checkin(worker, cumulative_ot_hours):
    """Stand-in for a real voice check-in. Fatigue climbs with overtime worked, at
    a per-worker rate. Production: return
    src.fatigue.predict_fatigue(live_wav, worker.id)['fatigue_score']."""
    rate = OT_FATIGUE_RATE.get(worker.id, 0.12)
    return round(min(1.0, rate * cumulative_ot_hours), 3)


def main():
    params = SchedParams()

    # --- deliberately short-staffed roster: base workers can't cover the evening ---
    workers = [
        Worker("W01", {"assembly", "packaging"}, willing_overtime=True),
        Worker("W02", {"assembly", "quality"},   willing_overtime=True),
        Worker("W03", {"packaging", "quality"},  willing_overtime=True),
        Worker("W04", {"assembly"},              willing_overtime=False),
        Worker("W05", {"quality", "packaging"},  willing_overtime=False),
    ]
    shifts = make_shifts()
    demand = base_demand()

    print("Overtime eligibility (fatigue gate):")
    for w in workers:
        print(f"  {w.id}: willing_OT={w.willing_overtime}, fatigue={w.fatigue:.2f} "
              f"-> ext_cap={ext_cap(w, params):.1f}h")

    # ---------- initial solve ----------
    sched, summ = build_and_solve(SchedulingProblem(workers, shifts, DEPARTMENTS, demand, params))
    show(sched, summ, "INITIAL SCHEDULE (overtime fills the staffing gap)")

    # ---------- reactive overtime loop ----------
    # Track cumulative overtime per worker; re-check every cycle; fatigued-out -> unavailable.
    cum_ot = {w.id: 0.0 for w in workers}
    for cycle in range(1, 4):
        ot_workers = sorted(set(sched[sched["overtime"]]["worker"])) if not sched.empty else []
        if not ot_workers:
            print(f"\n[cycle {cycle}] no one on overtime — schedule stable, stopping.")
            break

        print(f"\n[cycle {cycle}] voice check-ins for overtime workers: {ot_workers}")
        for w in workers:
            if w.id in ot_workers:
                cum_ot[w.id] += params.max_overtime_per_cycle
                w.fatigue = mock_checkin(w, cum_ot[w.id])
                fatigued = w.fatigue >= params.fatigue_ot_threshold
                if fatigued:
                    w.available = False
                print(f"    {w.id}: cum_OT={cum_ot[w.id]:.0f}h, fatigue={w.fatigue:.2f}"
                      + ("  -> FATIGUED, pulled off" if fatigued else "  -> OK, continues"))

        sched, summ = build_and_solve(SchedulingProblem(workers, shifts, DEPARTMENTS, demand, params))
        show(sched, summ, f"RE-SOLVED AFTER CYCLE {cycle}")

    print("\n✓ Reactive loop done. As workers fatigue out of overtime, the scheduler "
          "re-covers the work from whoever remains available (or reports unmet demand).")


if __name__ == "__main__":
    main()
