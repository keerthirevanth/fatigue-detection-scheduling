"""
Scheduler tests. Skipped automatically if gurobipy/license is unavailable so the
rest of the suite still runs in a bare CI.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scheduler.optimizer import (Worker, Shift, SchedulingProblem, SchedParams,
                                      build_and_solve, ext_cap)

gp = pytest.importorskip("gurobipy")   # skip whole module if Gurobi missing


def _shifts(day=1):
    return [
        Shift(f"d{day}_morning", day, "morning", 6.0, 8.0, False),
        Shift(f"d{day}_evening", day, "evening", 14.0, 8.0, False),
        Shift(f"d{day}_night",   day, "night",   22.0, 8.0, True),
    ]


def test_overtime_gated_by_fatigue():
    p = SchedParams()
    rested = Worker("A", {"x"}, willing_overtime=True, fatigue=0.0)
    tired  = Worker("B", {"x"}, willing_overtime=True, fatigue=0.7)   # above threshold
    unwill = Worker("C", {"x"}, willing_overtime=False, fatigue=0.0)
    gone   = Worker("D", {"x"}, willing_overtime=True, fatigue=0.0, available=False)
    assert ext_cap(rested, p) == p.max_overtime_per_cycle
    assert ext_cap(tired, p) == 0.0
    assert ext_cap(unwill, p) == 0.0
    assert ext_cap(gone, p) == 0.0


def test_coverage_met_with_ample_capacity():
    workers = [Worker(f"W{i}", {"assembly"}, willing_overtime=False) for i in range(6)]
    shifts = _shifts()
    demand = {(s.id, "assembly"): 8.0 for s in shifts}      # 1 worker each shift
    sched, summ = build_and_solve(SchedulingProblem(workers, shifts, ["assembly"], demand))
    assert summ["total_unmet_hours"] == 0.0
    assert not sched.empty


def test_overtime_fills_understaffed_shift():
    # 1 willing worker, demand = 12h in a shift -> needs 8h base + 4h overtime.
    workers = [Worker("W", {"assembly"}, willing_overtime=True, fatigue=0.0)]
    shifts = [Shift("d1_morning", 1, "morning", 6.0, 8.0, False)]
    demand = {("d1_morning", "assembly"): 12.0}
    sched, summ = build_and_solve(SchedulingProblem(workers, shifts, ["assembly"], demand))
    assert summ["total_overtime_hours"] > 0.0
    assert summ["total_unmet_hours"] == 0.0


def test_no_overtime_when_fatigued():
    # willing but already fatigued -> no overtime -> demand above 8h goes unmet.
    workers = [Worker("T", {"assembly"}, willing_overtime=True, fatigue=0.7)]
    shifts = [Shift("d1_morning", 1, "morning", 6.0, 8.0, False)]
    demand = {("d1_morning", "assembly"): 12.0}
    sched, summ = build_and_solve(SchedulingProblem(workers, shifts, ["assembly"], demand))
    assert summ["total_overtime_hours"] == 0.0
    assert summ["total_unmet_hours"] > 0.0
