"""
Deterministic tools for the agent layer. NO reasoning lives here — these
functions enforce the safety rules mechanically so no LLM output can bypass
them:

  * checkin_worker: the fatigue reading comes ONLY from state.checkin_provider
    (the sensor). At/above the threshold the worker is pulled automatically —
    the agent just gets told what happened.
  * commit_schedule: requires a human approver string; agents cannot self-approve.
  * set_worker_availability: can only RESTRICT; it can never re-activate a
    worker whose latest measured fatigue is at/above the threshold.

Every tool takes the shared PlantState first and returns a plain string (which
becomes the tool_result content shown to the model).
"""
import json

from src.scheduler.optimizer import SchedulingProblem, build_and_solve, ext_cap
from .state import PlantState


# --------------------------------------------------------------------- helpers
def _roster_text(schedule) -> str:
    if schedule is None or schedule.empty:
        return "(no assignments)"
    lines = []
    for _, r in schedule.iterrows():
        ot = f" +{r['overtime_hours']}h OT" if r["overtime"] else ""
        lines.append(f"{r['slot']:8s} | {r['department']:10s} | {r['worker']} "
                     f"({r['total_hours']}h{ot})")
    return "\n".join(lines)


# --------------------------------------------------------------------- tools
def get_roster_status(state: PlantState) -> str:
    """Read-only snapshot of workers, current schedule, and unmet demand."""
    workers = [{
        "id": w.id, "skills": sorted(w.skills), "available": w.available,
        "fatigue": w.fatigue, "willing_overtime": w.willing_overtime,
        "overtime_cap_hours": ext_cap(w, state.params),
    } for w in state.workers]
    out = {
        "workers": workers,
        "fatigue_ot_threshold": state.params.fatigue_ot_threshold,
        "demand": {f"{s}|{d}": h for (s, d), h in state.demand.items()},
        "current_schedule": _roster_text(state.current_schedule),
        "current_summary": state.current_summary or "no committed schedule yet",
        "has_pending_proposal": state.pending_proposal is not None,
        "recent_checkins": state.checkin_log[-10:],
    }
    return json.dumps(out, indent=1)


def solve_schedule(state: PlantState) -> str:
    """Run the Gurobi optimizer on the current state. Stores the result as a
    PENDING PROPOSAL — it does not become the live schedule until a human
    approves it via commit_schedule."""
    problem = SchedulingProblem(state.workers, state.shifts, state.departments,
                                state.demand, state.params)
    schedule, summary = build_and_solve(problem)
    state.pending_proposal = {"schedule": schedule, "summary": summary}
    state.log(f"solve_schedule: obj={summary['objective']}, "
              f"unmet={summary['total_unmet_hours']}h")
    return ("PROPOSAL (not yet committed — needs human approval):\n"
            + _roster_text(schedule) + "\n"
            + json.dumps({k: v for k, v in summary.items()
                          if k != "hours_per_worker"}, indent=1))


def commit_schedule(state: PlantState, approved_by: str) -> str:
    """Promote the pending proposal to the live schedule. GUARDRAIL: requires a
    named human approver — the agent may only call this after the human user
    has explicitly approved in conversation."""
    if not approved_by or not approved_by.strip():
        return "ERROR: commit refused — no human approver named. Ask the supervisor first."
    if state.pending_proposal is None:
        return "ERROR: nothing to commit — run solve_schedule first."
    state.current_schedule = state.pending_proposal["schedule"]
    state.current_summary = state.pending_proposal["summary"]
    state.pending_proposal = None
    state.log(f"commit_schedule: approved by {approved_by}")
    return f"Committed. Live schedule updated (approved by {approved_by})."


def checkin_worker(state: PlantState, worker_id: str, is_recheck: bool = False) -> str:
    """Run a voice check-in on a worker. The fatigue reading comes from the
    SENSOR (state.checkin_provider) — never from the caller. At/above the
    threshold the worker is pulled from further work automatically."""
    w = state.worker(worker_id)
    if w is None:
        return f"ERROR: unknown worker '{worker_id}'."
    reading = float(state.checkin_provider(worker_id))
    w.fatigue = reading
    fatigued = reading >= state.params.fatigue_ot_threshold
    if fatigued:
        w.available = False
    entry = {"worker": worker_id, "fatigue": round(reading, 3),
             "recheck": is_recheck, "outcome": "PULLED" if fatigued else "OK"}
    state.checkin_log.append(entry)
    state.log(f"checkin: {entry}")
    if fatigued:
        return (f"{worker_id}: fatigue={reading:.2f} >= threshold "
                f"{state.params.fatigue_ot_threshold} -> PULLED from work "
                "(now unavailable). The roster likely needs re-solving.")
    return f"{worker_id}: fatigue={reading:.2f} -> OK to continue."


def request_recheck(state: PlantState, worker_id: str) -> str:
    """Immediately repeat a check-in (for borderline readings). A recheck can
    confirm or worsen the assessment; it never grants extra overtime."""
    return checkin_worker(state, worker_id, is_recheck=True)


def set_worker_availability(state: PlantState, worker_id: str, available: bool,
                            reason: str) -> str:
    """Mark a worker available/unavailable (sick call, back from rest, pulled
    early as a precaution). GUARDRAIL: cannot re-activate a worker whose latest
    measured fatigue is at/above the threshold — only a fresh below-threshold
    check-in can do that."""
    w = state.worker(worker_id)
    if w is None:
        return f"ERROR: unknown worker '{worker_id}'."
    if available and w.fatigue >= state.params.fatigue_ot_threshold:
        return (f"ERROR: refused — {worker_id}'s last measured fatigue "
                f"({w.fatigue:.2f}) is at/above the threshold. They can only "
                "return via a fresh check-in below the threshold.")
    w.available = available
    state.log(f"availability: {worker_id} -> {available} ({reason})")
    return f"{worker_id} is now {'available' if available else 'unavailable'} ({reason})."


# --------------------------------------------------------------------- schemas
# JSON schemas the agents see. Note checkin_worker deliberately has NO fatigue
# parameter — the model can never supply a reading.
TOOL_SCHEMAS = {
    "get_roster_status": {
        "name": "get_roster_status",
        "description": "Get the current plant snapshot: every worker's skills, "
                       "availability, latest measured fatigue and overtime cap, "
                       "the demand table, the live schedule, and recent check-ins.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
    },
    "solve_schedule": {
        "name": "solve_schedule",
        "description": "Run the Gurobi shift optimizer on the current worker "
                       "availability and demand. Produces a PROPOSAL (not live "
                       "until a human approves). Use after anything changes: a "
                       "worker pulled for fatigue, a sick call, new demand.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
    },
    "commit_schedule": {
        "name": "commit_schedule",
        "description": "Promote the pending proposal to the live schedule. Only "
                       "call AFTER the human supervisor explicitly approved it in "
                       "conversation; pass their name.",
        "input_schema": {
            "type": "object",
            "properties": {"approved_by": {
                "type": "string",
                "description": "Name of the human who approved, e.g. 'Supervisor Jay'."}},
            "required": ["approved_by"], "additionalProperties": False,
        },
    },
    "checkin_worker": {
        "name": "checkin_worker",
        "description": "Run a voice fatigue check-in on one worker. The score "
                       "comes from the acoustic sensor — you cannot influence it. "
                       "At/above threshold the worker is pulled automatically.",
        "input_schema": {
            "type": "object",
            "properties": {"worker_id": {"type": "string"}},
            "required": ["worker_id"], "additionalProperties": False,
        },
    },
    "request_recheck": {
        "name": "request_recheck",
        "description": "Immediately repeat a worker's check-in when a reading is "
                       "borderline (within ~0.05 below the threshold). A recheck "
                       "can confirm or pull the worker; it never extends overtime.",
        "input_schema": {
            "type": "object",
            "properties": {"worker_id": {"type": "string"}},
            "required": ["worker_id"], "additionalProperties": False,
        },
    },
    "set_worker_availability": {
        "name": "set_worker_availability",
        "description": "Mark a worker available/unavailable (sick call, returned "
                       "from rest, precautionary early pull). Cannot re-activate "
                       "a worker whose measured fatigue is at/above threshold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "worker_id": {"type": "string"},
                "available": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["worker_id", "available", "reason"],
            "additionalProperties": False,
        },
    },
}

# Dispatch table used by the agent loop.
TOOL_FUNCTIONS = {
    "get_roster_status": get_roster_status,
    "solve_schedule": solve_schedule,
    "commit_schedule": commit_schedule,
    "checkin_worker": checkin_worker,
    "request_recheck": request_recheck,
    "set_worker_availability": set_worker_availability,
}
