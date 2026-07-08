"""
Fatigue-aware shift SCHEDULER (Gurobi MILP). The Operations-Research core.

Measurement-driven design (no fatigue prediction / no simulated dynamics):

  * A plant runs 3×8h shifts (Morning 06-14 / Evening 14-22 / Night 22-06) across
    several departments. The schedule assigns AVAILABLE workers to cover demand.
  * Primary goal: COMPLETE THE WORK — unmet demand is heavily penalised (soft).
  * OVERTIME: a willing worker may work past 8h. There is no fixed hour cap; what
    bounds overtime is REALITY — while on overtime the worker is voice-checked
    every few hours (src.fatigue.predict_fatigue). The moment a check-in reports
    fatigue, they stop and become unavailable, and we re-solve on who's left.
  * Fatigue therefore enters the model in ONE place: it GATES overtime. A worker
    whose latest check-in is fatigued (fatigue ≥ threshold) gets zero overtime.
    The base 8h shift is not checked (assumed rested).

Rolling horizon: solve → run one check-interval → feed in the new check-ins
(update `fatigue`/`available`) → re-solve. See scripts/demo_scheduler.py.

Model
-----
  decisions  x[w,s,d] ∈ {0,1}  worker w works base shift s in department d
             e[w,s,d] ≥ 0      overtime hours (only where x can be 1)
             u[s,d]   ≥ 0      unmet demand in worker-hours (soft)
  minimise   W_short·Σ u  +  W_ot·Σ e  +  W_fair·(Lmax − Lmin)
  s.t.       coverage, one base-shift/day, skills/availability,
             overtime ≤ fatigue-gated cap, fairness load bounds.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import pandas as pd


# --------------------------------------------------------------------- data
@dataclass
class Worker:
    id: str
    skills: Set[str]                 # departments this worker is qualified for
    willing_overtime: bool = False
    fatigue: float = 0.0             # latest measured fatigue ∈ [0,1] (only set while on OT)
    available: bool = True           # False once fatigued-out or resting


@dataclass
class Shift:
    id: str
    day: int
    slot: str                        # "morning" | "evening" | "night"
    clock_start: float               # wall-clock hour the shift begins
    length: float = 8.0
    is_night: bool = False


@dataclass
class SchedParams:
    max_overtime_per_cycle: float = 4.0   # OT granted per solve before the next check-in
    fatigue_ot_threshold: float = 0.6     # measured fatigue ≥ this → no overtime
    w_short: float = 1000.0               # unmet demand (primary — finish the work)
    w_overtime: float = 1.0               # overtime is mildly costly (prefer base hours)
    w_fair: float = 2.0                   # load-balancing across workers
    max_base_shifts_per_day: int = 1


@dataclass
class SchedulingProblem:
    workers: List[Worker]
    shifts: List[Shift]
    departments: List[str]
    demand: Dict[Tuple[str, str], float]   # (shift_id, dept) -> required worker-hours
    params: SchedParams = field(default_factory=SchedParams)


# --------------------------------------------------------------------- helpers
def ext_cap(worker: Worker, p: SchedParams) -> float:
    """Overtime a worker may take THIS cycle. Zero unless willing, available, and
    the latest check-in is below the fatigue threshold. This is the whole role of
    the fatigue model in the scheduler: a live gate on continued overtime."""
    if not worker.willing_overtime or not worker.available:
        return 0.0
    if worker.fatigue >= p.fatigue_ot_threshold:
        return 0.0
    return p.max_overtime_per_cycle


# --------------------------------------------------------------------- solve
def build_and_solve(problem: SchedulingProblem, verbose: bool = False):
    """Solve the rostering MILP. Returns (schedule_df, summary_dict)."""
    import gurobipy as gp
    from gurobipy import GRB

    W, S, D = problem.workers, problem.shifts, problem.departments
    p = problem.params
    demand = problem.demand
    shift_by_id = {s.id: s for s in S}

    m = gp.Model("fatigue_aware_roster")
    m.Params.OutputFlag = 1 if verbose else 0

    # x/e only where the worker is skilled for the department and currently available.
    def allowed(w: Worker, d: str) -> bool:
        return (d in w.skills) and w.available

    x, e = {}, {}
    for w in W:
        for s in S:
            for d in D:
                if allowed(w, d):
                    x[w.id, s.id, d] = m.addVar(vtype=GRB.BINARY, name=f"x_{w.id}_{s.id}_{d}")
                    e[w.id, s.id, d] = m.addVar(lb=0.0, name=f"e_{w.id}_{s.id}_{d}")

    u = {(s.id, d): m.addVar(lb=0.0, name=f"u_{s.id}_{d}") for s in S for d in D}

    # ---- constraints ----
    # (1) coverage — base + overtime worker-hours meet demand, minus soft shortfall
    for s in S:
        for d in D:
            supplied = gp.quicksum(s.length * x[w.id, s.id, d] + e[w.id, s.id, d]
                                   for w in W if (w.id, s.id, d) in x)
            m.addConstr(supplied + u[s.id, d] >= demand.get((s.id, d), 0.0),
                        name=f"cover_{s.id}_{d}")

    # (2) at most one department per shift
    for w in W:
        for s in S:
            terms = [x[w.id, s.id, d] for d in D if (w.id, s.id, d) in x]
            if terms:
                m.addConstr(gp.quicksum(terms) <= 1, name=f"one_dept_{w.id}_{s.id}")

    # (3) overtime only where assigned, capped by the live fatigue gate
    for w in W:
        cap = ext_cap(w, p)
        for s in S:
            for d in D:
                if (w.id, s.id, d) in e:
                    m.addConstr(e[w.id, s.id, d] <= cap * x[w.id, s.id, d],
                                name=f"otcap_{w.id}_{s.id}_{d}")

    # (4) one BASE shift per worker per day (overtime extends that shift, not a new one)
    days = sorted({s.day for s in S})
    for w in W:
        for day in days:
            terms = [x[w.id, s.id, d] for s in S if s.day == day
                     for d in D if (w.id, s.id, d) in x]
            if terms:
                m.addConstr(gp.quicksum(terms) <= p.max_base_shifts_per_day,
                            name=f"perday_{w.id}_{day}")

    # (5) fairness — minimise the spread between busiest and least-busy worker
    load = {w.id: gp.quicksum(s.length * x[w.id, s.id, d] + e[w.id, s.id, d]
                              for s in S for d in D if (w.id, s.id, d) in x) for w in W}
    Lmax = m.addVar(lb=0.0, name="Lmax")
    Lmin = m.addVar(lb=0.0, name="Lmin")
    for w in W:
        m.addConstr(Lmax >= load[w.id])
        m.addConstr(Lmin <= load[w.id])

    # ---- objective ----
    m.setObjective(p.w_short * gp.quicksum(u.values())
                   + p.w_overtime * gp.quicksum(e.values())
                   + p.w_fair * (Lmax - Lmin), GRB.MINIMIZE)

    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Solver status {m.Status} — no usable solution.")

    # ---- extract schedule ----
    rows = []
    for (w_id, s_id, d), var in x.items():
        if var.X > 0.5:
            s = shift_by_id[s_id]
            ot = e[w_id, s_id, d].X
            rows.append({
                "worker": w_id, "day": s.day, "shift": s_id, "slot": s.slot,
                "department": d, "base_hours": s.length,
                "overtime_hours": round(ot, 2), "total_hours": round(s.length + ot, 2),
                "overtime": ot > 1e-6,
            })
    schedule = (pd.DataFrame(rows).sort_values(["day", "slot", "department"])
                if rows else pd.DataFrame())

    summary = {
        "objective": round(m.ObjVal, 2),
        "total_unmet_hours": round(sum(v.X for v in u.values()), 2),
        "unmet_by_slot": {k: round(v.X, 2) for k, v in u.items() if v.X > 1e-6},
        "total_overtime_hours": round(sum(v.X for v in e.values()), 2),
        "hours_per_worker": {w.id: round(load[w.id].getValue(), 2) for w in W},
        "workers_used": schedule["worker"].nunique() if not schedule.empty else 0,
    }
    return schedule, summary
