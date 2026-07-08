"""
The three agents (Anthropic tool-use, manual agentic loop).

  Orchestrator  — talks to the human supervisor; delegates to the two
                  sub-agents (agents-as-tools pattern); owns commit approval.
  Scheduling    — runs the Gurobi solver, diagnoses unmet demand, proposes options.
  Monitoring    — runs voice check-ins on overtime workers, judges borderline
                  readings (conservative-only), reports pulls.

Safety model: the LLMs coordinate and explain. The hard rules (fatigue
threshold, human approval, no self-supplied readings) are enforced in
src/agent/tools.py, not in prompts — a prompt injection cannot bypass them.
"""
import os
from dataclasses import dataclass, field
from typing import List

import anthropic

from .state import PlantState
from .tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

DEFAULT_MODEL = os.environ.get("FATIGUE_AGENT_MODEL", "claude-opus-4-8")
MAX_AGENT_TURNS = 12          # hard stop on any single agent's tool loop


# --------------------------------------------------------------------- prompts
ORCHESTRATOR_SYSTEM = """\
You are the shift-operations assistant for an industrial plant that runs three
8-hour shifts (morning/evening/night) across several departments. Workers may
take overtime; while on overtime they are voice-checked for fatigue every few
hours, and the measured score gates whether they may continue.

You are the supervisor's single point of contact. You coordinate two
specialists and report back clearly and briefly:
  * ask_scheduling_agent — anything about building/re-building the roster,
    coverage, unmet demand, overtime allocation.
  * ask_monitoring_agent — anything about fatigue check-ins, rechecks, or
    pulling workers.
Give each specialist a self-contained task description; they don't see this
conversation.

Hard rules you must follow:
  * You never invent fatigue scores or worker data — everything comes from tools.
  * Schedule changes are PROPOSALS until the human approves. Only call
    commit_schedule after the supervisor has explicitly approved in this
    conversation, and pass their name. Never invent approval.
  * You cannot mark a fatigued worker as fit — only a fresh below-threshold
    check-in can do that, and only the sensor produces readings.
When reporting, lead with the outcome (covered or not, who was pulled), then
the key numbers. Flag any unmet demand explicitly — never gloss over it."""

SCHEDULING_SYSTEM = """\
You are the scheduling specialist for an industrial plant (3×8h shifts,
multiple departments, overtime gated by measured fatigue). You receive one
self-contained task, do the work with your tools, and reply with a compact
report.

Method: check get_roster_status first if you need context; use solve_schedule
to produce a proposal. If demand is unmet, DIAGNOSE why (who is unavailable,
which skills are short, whose overtime is fatigue-capped) and present the
realistic options (approve overtime for someone willing+rested, accept the
shortfall, redistribute across departments). You cannot commit schedules —
that needs human approval upstream — and you cannot change fatigue values or
re-activate fatigued workers. Never invent workers or data."""

MONITORING_SYSTEM = """\
You are the fatigue-monitoring specialist. You receive one self-contained task,
run the needed voice check-ins with your tools, and reply with a compact report.

The sensor and the threshold decide the outcome of a check-in — you cannot
override them, and workers at/above threshold are pulled automatically. Your
judgment applies only to BORDERLINE readings (within about 0.05 below the
threshold): for those, use request_recheck, and if still borderline you may
pull the worker early via set_worker_availability as a precaution.
Conservative-only rule: you may pull a worker EARLIER than the threshold; you
may never keep one working past it, and you may never mark a fatigued worker
fit. Report every reading and every pull explicitly."""


# --------------------------------------------------------------------- config
@dataclass
class AgentSpec:
    name: str
    system: str
    tool_names: List[str]


SCHEDULING_AGENT = AgentSpec(
    name="scheduling",
    system=SCHEDULING_SYSTEM,
    tool_names=["get_roster_status", "solve_schedule", "set_worker_availability"],
)

MONITORING_AGENT = AgentSpec(
    name="monitoring",
    system=MONITORING_SYSTEM,
    tool_names=["get_roster_status", "checkin_worker", "request_recheck",
                "set_worker_availability"],
)

# The orchestrator's own deterministic tools. It deliberately CANNOT call
# solve_schedule or checkin_worker itself — it must delegate, keeping each
# responsibility with one agent.
ORCHESTRATOR_TOOL_NAMES = ["get_roster_status", "commit_schedule"]

_DELEGATION_SCHEMAS = [
    {
        "name": "ask_scheduling_agent",
        "description": "Delegate a scheduling task (build/re-solve the roster, "
                       "diagnose coverage, overtime options) to the scheduling "
                       "specialist. Provide a fully self-contained task.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"], "additionalProperties": False,
        },
    },
    {
        "name": "ask_monitoring_agent",
        "description": "Delegate a fatigue-monitoring task (check-ins, rechecks, "
                       "precautionary pulls) to the monitoring specialist. "
                       "Provide a fully self-contained task.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"], "additionalProperties": False,
        },
    },
]


# --------------------------------------------------------------------- loop
def _run_tool_loop(client, model, system, tools, messages, state: PlantState,
                   on_event=None, extra_dispatch=None):
    """Manual agentic loop: call the model, execute tool calls against the
    shared state, feed results back, repeat until the model stops. Returns the
    final assistant text; `messages` is mutated in place (full history kept)."""
    notify = on_event or (lambda kind, detail: None)

    for _ in range(MAX_AGENT_TURNS):
        response = client.messages.create(
            model=model, max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system, tools=tools, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return next((b.text for b in response.content if b.type == "text"), "")

        # Execute every tool call; ALL results go back in ONE user message.
        results = []
        for tu in tool_uses:
            notify("tool", f"{tu.name}({json_compact(tu.input)})")
            try:
                if extra_dispatch and tu.name in extra_dispatch:
                    output = extra_dispatch[tu.name](**tu.input)
                else:
                    output = TOOL_FUNCTIONS[tu.name](state, **tu.input)
                is_error = output.startswith("ERROR")
            except Exception as exc:                    # tool bug ≠ crash the loop
                output, is_error = f"ERROR: tool raised {exc!r}", True
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    return "(agent stopped: reached the tool-call turn limit)"


def json_compact(obj) -> str:
    import json
    s = json.dumps(obj, default=str)
    return s if len(s) <= 120 else s[:117] + "..."


def run_subagent(client, model, spec: AgentSpec, task: str, state: PlantState,
                 on_event=None) -> str:
    """One-shot worker agent: fresh context, gets a task string, returns a report."""
    tools = [TOOL_SCHEMAS[n] for n in spec.tool_names]
    messages = [{"role": "user", "content": task}]
    if on_event:
        on_event("delegate", f"→ {spec.name} agent: {task[:100]}")
    report = _run_tool_loop(client, model, spec.system, tools, messages, state,
                            on_event=on_event)
    if on_event:
        on_event("report", f"← {spec.name} agent replied ({len(report)} chars)")
    return report


class FatigueOpsAgent:
    """The orchestrator — the one agent the human talks to. Maintains the
    conversation across turns; sub-agents are invoked as tools with fresh
    context each time."""

    def __init__(self, state: PlantState, model: str = DEFAULT_MODEL,
                 client: anthropic.Anthropic = None, on_event=None):
        self.state = state
        self.model = model
        self.client = client or anthropic.Anthropic()
        self.on_event = on_event
        self.messages: list = []

        self._tools = ([TOOL_SCHEMAS[n] for n in ORCHESTRATOR_TOOL_NAMES]
                       + _DELEGATION_SCHEMAS)
        self._dispatch = {
            "ask_scheduling_agent": lambda task: run_subagent(
                self.client, self.model, SCHEDULING_AGENT, task, self.state,
                self.on_event),
            "ask_monitoring_agent": lambda task: run_subagent(
                self.client, self.model, MONITORING_AGENT, task, self.state,
                self.on_event),
        }

    def chat(self, user_message: str) -> str:
        """Send one supervisor message; returns the orchestrator's reply."""
        self.messages.append({"role": "user", "content": user_message})
        return _run_tool_loop(self.client, self.model, ORCHESTRATOR_SYSTEM,
                              self._tools, self.messages, self.state,
                              on_event=self.on_event,
                              extra_dispatch=self._dispatch)
