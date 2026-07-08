"""
Public entry point for the agent layer.

    from src.agent.orchestrator import FatigueOpsAgent, PlantState

    agent = FatigueOpsAgent(state)
    print(agent.chat("Build tonight's schedule — can we cover demand?"))

Architecture: one Orchestrator (talks to the human, owns commit approval) and
two one-shot specialists it invokes as tools — a Scheduling agent (Gurobi) and
a Monitoring agent (voice check-ins). Hard safety rules live in tools.py, not
in prompts. See agents.py for the loop and system prompts.
"""
from .state import PlantState
from .agents import (FatigueOpsAgent, run_subagent,
                     SCHEDULING_AGENT, MONITORING_AGENT, DEFAULT_MODEL)

__all__ = ["FatigueOpsAgent", "PlantState", "run_subagent",
           "SCHEDULING_AGENT", "MONITORING_AGENT", "DEFAULT_MODEL"]
