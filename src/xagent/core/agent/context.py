import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """
    A shared context during agent execution, useful for multi-step coordination.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    history: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
