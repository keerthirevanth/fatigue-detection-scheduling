"""
Synthetic multi-week workforce scenario generator + a GROUND-TRUTH fatigue
process used only by the simulation to (a) generate what a voice check-in would
measure, with noise, and (b) let the evaluator judge whether a policy assigned
an already-fatigued worker — even the baseline, which never measures fatigue and
so can't know it did this. Neither policy is ever handed the ground truth
directly; the fatigue-aware policy only ever sees a noisy "check-in" reading,
consistent with the measurement-driven design of the production scheduler.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

DEPARTMENTS = ["assembly", "packaging", "quality"]
SLOTS = ["morning", "evening", "night"]
BASE_SHIFT_HOURS = 8.0
FATIGUE_DANGER_THRESHOLD = 0.6   # ground-truth safety line (matches the policy's own gate)


@dataclass
class WorkerProfile:
    id: str
    skills: Set[str]
    willing_overtime: bool
    build_rate: float      # how fast THIS worker's true fatigue rises per OT hour
    recovery_rate: float   # how fast it falls on a rest day
    true_fatigue: float = 0.25   # ground truth; NEVER given directly to a policy


@dataclass
class DayScenario:
    day: int
    demand: Dict[Tuple[str, str], float]     # (slot, dept) -> required worker-hours
    sicked_out: Set[str]                     # worker ids unavailable today


@dataclass
class Scenario:
    workers: List[WorkerProfile]
    days: List[DayScenario]
    departments: List[str] = field(default_factory=lambda: list(DEPARTMENTS))


def generate_workforce(n_workers=18, seed=0) -> List[WorkerProfile]:
    rng = np.random.default_rng(seed)
    workers = []
    for i in range(n_workers):
        n_skills = rng.choice([1, 2], p=[0.4, 0.6])
        skills = set(rng.choice(DEPARTMENTS, size=n_skills, replace=False))
        workers.append(WorkerProfile(
            id=f"W{i+1:02d}", skills=skills,
            willing_overtime=bool(rng.random() < 0.7),
            build_rate=float(rng.uniform(0.05, 0.14)),     # individual variation
            recovery_rate=float(rng.uniform(0.20, 0.35)),
            true_fatigue=float(rng.uniform(0.15, 0.30)),   # start rested-ish
        ))
    return workers


def _base_demand(rng, spike: bool) -> Dict[Tuple[str, str], float]:
    d = {}
    for slot in SLOTS:
        for dept, base in (("assembly", 24.0), ("packaging", 12.0), ("quality", 12.0)):
            mult = rng.uniform(1.3, 1.6) if spike else rng.uniform(0.9, 1.1)
            d[(slot, dept)] = round(base * mult, 1)
    return d


def generate_scenario(n_workers=18, n_days=21, seed=0, spike_prob=0.15,
                      sickout_prob=0.05) -> Scenario:
    """A multi-week scenario: daily demand (occasional surge days) and random
    single-day sick-outs, on a fixed workforce."""
    rng = np.random.default_rng(seed)
    workers = generate_workforce(n_workers, seed=seed)

    days = []
    for d in range(n_days):
        spike = rng.random() < spike_prob
        demand = _base_demand(rng, spike)
        sicked = {w.id for w in workers if rng.random() < sickout_prob}
        days.append(DayScenario(day=d, demand=demand, sicked_out=sicked))
    return Scenario(workers=workers, days=days)


def update_true_fatigue(worker: WorkerProfile, hours_worked: float) -> float:
    """One day's ground-truth fatigue update. Working (esp. overtime) raises it;
    a light/no-work day lets it decay. This is the simulation's oracle — never
    exposed to either scheduling policy, only used to score outcomes."""
    overtime = max(0.0, hours_worked - BASE_SHIFT_HOURS)
    if hours_worked <= 1e-6:                      # day off / not scheduled
        new = worker.true_fatigue * (1 - worker.recovery_rate)
    else:
        rise = worker.build_rate * (1.0 + overtime)   # overtime disproportionately taxing
        new = worker.true_fatigue + rise * (1 - worker.true_fatigue)
        new -= worker.recovery_rate * 0.15             # light natural recovery even on a work day
    return float(np.clip(new, 0.0, 1.0))


def measure_checkin(worker: WorkerProfile, noise_std=0.05, seed=None) -> float:
    """What a voice check-in would read for this worker today: ground truth
    plus sensor noise (stand-in for real acoustic-model error)."""
    rng = np.random.default_rng(seed)
    return float(np.clip(worker.true_fatigue + rng.normal(0, noise_std), 0.0, 1.0))
