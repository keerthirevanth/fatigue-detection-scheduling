"""
Demo of the 3-agent layer: a supervisor conversation over a short-staffed plant.

    python scripts/demo_agents.py

Needs Anthropic API credentials (ANTHROPIC_API_KEY env var, or an `ant auth
login` profile). Costs a few cents per run (several claude-opus-4-8 calls; set
FATIGUE_AGENT_MODEL to override the model).

The voice check-in is MOCKED here (fatigue rises with cumulative overtime) but
flows through the exact same checkin_worker tool the real system would use —
swap the provider for src.fatigue.predict_fatigue on live audio in production.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import anthropic

from src.scheduler.optimizer import Worker, Shift, SchedParams
from src.agent.orchestrator import FatigueOpsAgent, PlantState

DEPARTMENTS = ["assembly", "packaging", "quality"]
SLOTS = [("morning", 6.0), ("evening", 14.0), ("night", 22.0)]


def build_state() -> PlantState:
    workers = [
        Worker("W01", {"assembly", "packaging"}, willing_overtime=True),
        Worker("W02", {"assembly", "quality"},   willing_overtime=True),
        Worker("W03", {"packaging", "quality"},  willing_overtime=True),
        Worker("W04", {"assembly"},              willing_overtime=False),
        Worker("W05", {"quality", "packaging"},  willing_overtime=False),
    ]
    shifts = [Shift(f"d1_{slot}", 1, slot, clk, 8.0, slot == "night")
              for slot, clk in SLOTS]
    demand = {}
    for slot, _ in SLOTS:
        sid = f"d1_{slot}"
        demand[(sid, "assembly")] = 16.0
        demand[(sid, "packaging")] = 8.0
        demand[(sid, "quality")] = 8.0

    # --- mock sensor: fatigue rises with cumulative overtime hours -----------
    # PRODUCTION SWAP-IN POINT: replace the body with
    #   predict_fatigue(record_audio(worker_id), worker_id, model, baselines)["fatigue_score"]
    cum_ot = {w.id: 0.0 for w in workers}
    rate = {"W01": 0.10, "W02": 0.16, "W03": 0.12}

    def mock_checkin(worker_id: str) -> float:
        cum_ot[worker_id] = cum_ot.get(worker_id, 0.0) + 4.0
        return min(1.0, rate.get(worker_id, 0.12) * cum_ot[worker_id])

    return PlantState(workers=workers, shifts=shifts, departments=DEPARTMENTS,
                      demand=demand, checkin_provider=mock_checkin,
                      params=SchedParams())


def main():
    state = build_state()

    def on_event(kind, detail):
        print(f"      [{kind}] {detail}")

    try:
        agent = FatigueOpsAgent(state, on_event=on_event)
        supervisor_turns = [
            "Build tonight's schedule. Can we cover demand, and who is on overtime?",
            "A few hours in — run fatigue check-ins on everyone currently on "
            "overtime and re-plan around anyone who gets pulled.",
            "Understood. I approve the latest proposal — commit it. "
            "Approved by Supervisor Jay.",
        ]
        for turn in supervisor_turns:
            print(f"\n{'='*74}\nSUPERVISOR: {turn}\n{'-'*74}")
            reply = agent.chat(turn)
            print(f"ASSISTANT:\n{reply}")

        print(f"\n{'='*74}\nAUDIT LOG (what the tools actually did):")
        for line in state.audit_log:
            print(f"  • {line}")

    except anthropic.AuthenticationError:
        print("\nNo valid Anthropic credentials found.")
        print("Set ANTHROPIC_API_KEY, or run `ant auth login`, then re-run.")
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("\nCould not reach the Anthropic API — check your connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
