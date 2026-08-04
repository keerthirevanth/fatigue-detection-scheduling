"""
Two scheduling policies run over the SAME synthetic scenario, so their outcomes
are directly comparable:

  baseline_policy        "traditional" practice — fixed-rotation greedy
                          assignment, flat overtime cap, NO fatigue awareness.
  fatigue_aware_policy    your system — the Gurobi MILP from
                          src.scheduler.optimizer, with each worker's overtime
                          gated by a same-day voice check-in reading.

Both run day-by-day. Each policy maintains its own independent copy of every
worker's ground-truth fatigue (scenario.update_true_fatigue), because the two
policies make different assignments and so the two "worlds" diverge — that
divergence is the whole point of the comparison.
"""
import copy
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .scenario import (Scenario, WorkerProfile, SLOTS, BASE_SHIFT_HOURS,
                       FATIGUE_DANGER_THRESHOLD, update_true_fatigue, measure_checkin)
from src.scheduler.optimizer import (Worker, Shift, SchedulingProblem, SchedParams,
                                      build_and_solve)

BASELINE_FLAT_OT_CAP = 4.0   # traditional practice: allow up to base+4h, no fatigue check


@dataclass
class DayResult:
    day: int
    hours_worked: Dict[str, float]           # worker_id -> total hours today
    overtime_hours: Dict[str, float]         # worker_id -> OT hours today
    unmet_demand_hours: float
    unsafe_overtime_assignments: List[str]   # worker ids given OT while true-fatigued


@dataclass
class PolicyRun:
    name: str
    day_results: List[DayResult] = field(default_factory=list)

    # ---- aggregate KPIs -----------------------------------------------------
    def total_unsafe_assignments(self) -> int:
        return sum(len(d.unsafe_overtime_assignments) for d in self.day_results)

    def total_unmet_hours(self) -> float:
        return sum(d.unmet_demand_hours for d in self.day_results)

    def total_overtime_hours(self) -> float:
        return sum(sum(d.overtime_hours.values()) for d in self.day_results)

    def total_demand_hours(self, scenario: Scenario) -> float:
        return sum(sum(day.demand.values()) for day in scenario.days)

    def coverage_rate(self, scenario: Scenario) -> float:
        total = self.total_demand_hours(scenario)
        return 1.0 - (self.total_unmet_hours() / total if total else 0.0)

    def fairness_stdev(self) -> float:
        """Std-dev of each worker's total hours over the whole horizon — lower
        is fairer (workload spread more evenly)."""
        totals: Dict[str, float] = {}
        for d in self.day_results:
            for wid, h in d.hours_worked.items():
                totals[wid] = totals.get(wid, 0.0) + h
        return float(np.std(list(totals.values()))) if totals else 0.0


# --------------------------------------------------------------------- baseline
def baseline_policy(scenario: Scenario) -> PolicyRun:
    workers = copy.deepcopy(scenario.workers)
    by_id = {w.id: w for w in workers}
    cumulative_hours = {w.id: 0.0 for w in workers}
    run = PolicyRun(name="baseline_fixed_rotation")

    for day in scenario.days:
        available = [w for w in workers if w.id not in day.sicked_out]
        hours_today: Dict[str, float] = {w.id: 0.0 for w in workers}
        ot_today: Dict[str, float] = {w.id: 0.0 for w in workers}
        assigned_today = set()
        unmet = 0.0
        unsafe = []

        for slot in SLOTS:
            for dept in scenario.departments:
                remaining = day.demand.get((slot, dept), 0.0)
                # rotation: least-worked-so-far, skilled, available, not already
                # committed to a different shift today
                candidates = sorted(
                    (w for w in available if dept in w.skills and w.id not in assigned_today),
                    key=lambda w: cumulative_hours[w.id])

                for w in candidates:
                    if remaining <= 1e-6:
                        break
                    base = min(BASE_SHIFT_HOURS, remaining)
                    ot = 0.0
                    still_short = remaining - base
                    # flat cap: allow overtime up to the legal limit, NO fatigue check
                    if still_short > 1e-6 and w.willing_overtime:
                        ot = min(BASELINE_FLAT_OT_CAP, still_short)
                    hours = base + ot
                    hours_today[w.id] = hours
                    ot_today[w.id] = ot
                    assigned_today.add(w.id)
                    cumulative_hours[w.id] += hours
                    remaining -= hours
                    if ot > 1e-6 and w.true_fatigue >= FATIGUE_DANGER_THRESHOLD:
                        unsafe.append(w.id)   # baseline never knew — evaluator does
                unmet += max(0.0, remaining)

        for w in workers:
            w.true_fatigue = update_true_fatigue(w, hours_today[w.id])

        run.day_results.append(DayResult(day.day, hours_today, ot_today, unmet, unsafe))
    return run


# --------------------------------------------------------------- fatigue-aware
def fatigue_aware_policy(scenario: Scenario, checkin_noise=0.05,
                         base_seed=1000) -> PolicyRun:
    workers = copy.deepcopy(scenario.workers)
    run = PolicyRun(name="fatigue_aware_gurobi")

    for day in scenario.days:
        # today's voice check-in reading for every worker (noisy, not ground truth)
        readings = {w.id: measure_checkin(w, checkin_noise, seed=base_seed + day.day * 97 + i)
                   for i, w in enumerate(workers)}

        opt_workers = [
            Worker(id=w.id, skills=set(w.skills), willing_overtime=w.willing_overtime,
                   fatigue=readings[w.id], available=(w.id not in day.sicked_out))
            for w in workers
        ]
        shifts = [Shift(id=f"d{day.day}_{slot}", day=day.day, slot=slot,
                        clock_start={"morning": 6.0, "evening": 14.0, "night": 22.0}[slot],
                        length=BASE_SHIFT_HOURS, is_night=(slot == "night"))
                 for slot in SLOTS]
        demand = {(f"d{day.day}_{slot}", dept): hrs
                 for (slot, dept), hrs in day.demand.items()}

        schedule, summary = build_and_solve(
            SchedulingProblem(opt_workers, shifts, scenario.departments, demand, SchedParams()))

        hours_today = {w.id: 0.0 for w in workers}
        ot_today = {w.id: 0.0 for w in workers}
        unsafe = []
        if not schedule.empty:
            for _, r in schedule.iterrows():
                hours_today[r["worker"]] = r["total_hours"]
                ot_today[r["worker"]] = r["overtime_hours"]
                if r["overtime_hours"] > 1e-6:
                    w = next(w for w in workers if w.id == r["worker"])
                    if w.true_fatigue >= FATIGUE_DANGER_THRESHOLD:
                        # the check-in reading MISSED it (sensor noise) — should be rare
                        unsafe.append(w.id)

        for w in workers:
            w.true_fatigue = update_true_fatigue(w, hours_today[w.id])

        run.day_results.append(DayResult(day.day, hours_today, ot_today,
                                         summary["total_unmet_hours"], unsafe))
    return run


def summarize(run: PolicyRun, scenario: Scenario) -> dict:
    return {
        "policy": run.name,
        "days": len(run.day_results),
        "total_demand_hours": round(run.total_demand_hours(scenario), 1),
        "unmet_demand_hours": round(run.total_unmet_hours(), 1),
        "coverage_rate": round(run.coverage_rate(scenario), 4),
        "total_overtime_hours": round(run.total_overtime_hours(), 1),
        "unsafe_overtime_assignments": run.total_unsafe_assignments(),
        "fairness_stdev_hours": round(run.fairness_stdev(), 2),
    }
