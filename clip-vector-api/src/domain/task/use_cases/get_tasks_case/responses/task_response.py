"""
Task response model.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass  
class TaskResponse:
    """
    Response model for a task.
    """
    id: str
    title: Optional[str]
    description: str
    is_completed: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
