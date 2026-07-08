"""
PlantState — the single shared "world state" for the agent layer.

Every tool reads and mutates THIS object; agents themselves hold no state.
That prevents the classic multi-agent bug where two agents act on different
views of the world.

Key design point (safety): `checkin_provider` is the ONLY source of fatigue
readings. It is a callback wired in by the host application — in production it
records audio and calls src.fatigue.predict_fatigue; in the demo it is a mock.
An LLM agent can *request* a check-in but can never *supply* the score.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.scheduler.optimizer import Worker, Shift, SchedParams


@dataclass
class PlantState:
    workers: List[Worker]
    shifts: List[Shift]
    departments: List[str]
    demand: Dict[Tuple[str, str], float]          # (shift_id, dept) -> worker-hours
    checkin_provider: Callable[[str], float]      # worker_id -> measured fatigue [0,1]
    params: SchedParams = field(default_factory=SchedParams)

    # Solver outputs. A solve produces a PROPOSAL; only commit_schedule (with a
    # human approver) promotes it to the current schedule.
    current_schedule: Optional[pd.DataFrame] = None
    current_summary: Optional[dict] = None
    pending_proposal: Optional[dict] = None       # {"schedule": df, "summary": dict}

    checkin_log: List[dict] = field(default_factory=list)
    audit_log: List[str] = field(default_factory=list)

    def worker(self, worker_id: str) -> Optional[Worker]:
        return next((w for w in self.workers if w.id == worker_id), None)

    def log(self, msg: str) -> None:
        self.audit_log.append(msg)
