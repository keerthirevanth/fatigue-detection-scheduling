"""
Simulation sanity tests. The baseline policy needs no solver; the fatigue-aware
policy needs gurobipy — those tests are skipped automatically if unavailable.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.scenario import (generate_scenario, update_true_fatigue,
                                     measure_checkin, WorkerProfile,
                                     FATIGUE_DANGER_THRESHOLD)
from src.simulation.policies import baseline_policy, fatigue_aware_policy, summarize


def test_scenario_generation_deterministic():
    s1 = generate_scenario(n_workers=6, n_days=4, seed=42)
    s2 = generate_scenario(n_workers=6, n_days=4, seed=42)
    assert [w.true_fatigue for w in s1.workers] == [w.true_fatigue for w in s2.workers]
    assert len(s1.days) == 4 and len(s1.workers) == 6


def test_true_fatigue_rises_with_overtime_and_falls_on_rest():
    w = WorkerProfile("A", {"assembly"}, True, build_rate=0.1, recovery_rate=0.3,
                      true_fatigue=0.3)
    worked_hard = update_true_fatigue(w, hours_worked=12.0)   # 8 base + 4 OT
    assert worked_hard > w.true_fatigue

    w2 = WorkerProfile("B", {"assembly"}, True, build_rate=0.1, recovery_rate=0.3,
                       true_fatigue=0.6)
    rested = update_true_fatigue(w2, hours_worked=0.0)
    assert rested < w2.true_fatigue


def test_checkin_noisy_but_centered_on_truth():
    w = WorkerProfile("A", {"assembly"}, True, 0.1, 0.3, true_fatigue=0.5)
    readings = [measure_checkin(w, noise_std=0.05, seed=i) for i in range(50)]
    assert abs(sum(readings) / len(readings) - 0.5) < 0.03   # centered, low noise


def test_baseline_never_gates_on_fatigue():
    """The baseline must be capable of unsafe assignments — otherwise the
    comparison is meaningless. On a scenario forced to be short-staffed, at
    least some seeds should show unsafe overtime assignments."""
    found_unsafe = False
    for seed in range(5):
        sc = generate_scenario(n_workers=6, n_days=10, seed=seed, spike_prob=0.4)
        run = baseline_policy(sc)
        if run.total_unsafe_assignments() > 0:
            found_unsafe = True
            break
    assert found_unsafe, "expected at least one seed to produce an unsafe baseline assignment"


def test_fatigue_aware_reduces_unsafe_assignments():
    pytest.importorskip("gurobipy")
    sc = generate_scenario(n_workers=10, n_days=10, seed=1, spike_prob=0.3)
    base = baseline_policy(sc)
    aware = fatigue_aware_policy(sc)
    # fatigue-aware must not do WORSE on safety than the baseline
    assert aware.total_unsafe_assignments() <= base.total_unsafe_assignments()


def test_summarize_fields():
    sc = generate_scenario(n_workers=6, n_days=5, seed=2)
    run = baseline_policy(sc)
    s = summarize(run, sc)
    for key in ("coverage_rate", "unsafe_overtime_assignments", "total_overtime_hours",
               "fairness_stdev_hours", "unmet_demand_hours"):
        assert key in s
    assert 0.0 <= s["coverage_rate"] <= 1.0
