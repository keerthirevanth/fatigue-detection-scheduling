"""
Agent-layer tests — deterministic tools + guardrails only. No API key needed:
these never call an LLM; they verify that the safety rules hold no matter what
an agent (or a prompt injection) tries. Gurobi-dependent tests are skipped if
gurobipy is missing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scheduler.optimizer import Worker, Shift, SchedParams
from src.agent.state import PlantState
from src.agent import tools


def _state(checkin_values=None):
    """Small plant: 2 workers, 1 shift, sensor readings supplied per test."""
    values = dict(checkin_values or {})

    def provider(worker_id):                    # the ONLY source of readings
        return values.get(worker_id, 0.2)

    return PlantState(
        workers=[
            Worker("A", {"assembly"}, willing_overtime=True),
            Worker("B", {"assembly"}, willing_overtime=True),
        ],
        shifts=[Shift("d1_morning", 1, "morning", 6.0, 8.0, False)],
        departments=["assembly"],
        demand={("d1_morning", "assembly"): 8.0},
        checkin_provider=provider,
        params=SchedParams(),
    )


# ------------------------------------------------------------- check-in gate
def test_checkin_pulls_fatigued_worker_mechanically():
    st = _state({"A": 0.75})                    # sensor says fatigued
    out = tools.checkin_worker(st, "A")
    assert "PULLED" in out
    assert st.worker("A").available is False
    assert st.checkin_log[-1]["outcome"] == "PULLED"


def test_checkin_ok_below_threshold():
    st = _state({"A": 0.30})
    out = tools.checkin_worker(st, "A")
    assert "OK" in out
    assert st.worker("A").available is True


def test_agent_cannot_reactivate_fatigued_worker():
    st = _state({"A": 0.9})
    tools.checkin_worker(st, "A")               # pulled
    out = tools.set_worker_availability(st, "A", True, "agent says fine")
    assert out.startswith("ERROR")              # guardrail refuses
    assert st.worker("A").available is False


def test_precautionary_early_pull_is_allowed():
    st = _state({"A": 0.55})                    # borderline, below threshold
    tools.checkin_worker(st, "A")
    out = tools.set_worker_availability(st, "A", False, "borderline, precaution")
    assert not out.startswith("ERROR")
    assert st.worker("A").available is False


# ------------------------------------------------------------- commit gate
def test_commit_requires_pending_proposal_and_approver():
    st = _state()
    assert tools.commit_schedule(st, "Supervisor Jay").startswith("ERROR")  # nothing solved
    st.pending_proposal = {"schedule": None, "summary": {"objective": 0}}
    assert tools.commit_schedule(st, "").startswith("ERROR")               # no approver
    assert tools.commit_schedule(st, "   ").startswith("ERROR")            # blank approver
    out = tools.commit_schedule(st, "Supervisor Jay")
    assert "Committed" in out
    assert st.pending_proposal is None


# ------------------------------------------------------------- solver tool
def test_solve_creates_pending_proposal_not_live_schedule():
    pytest.importorskip("gurobipy")
    st = _state()
    out = tools.solve_schedule(st)
    assert "PROPOSAL" in out
    assert st.pending_proposal is not None
    assert st.current_schedule is None          # not live until committed
