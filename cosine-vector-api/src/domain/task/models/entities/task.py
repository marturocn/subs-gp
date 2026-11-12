"""
Domain Task entity.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """
    Domain Task entity.
    Represents a task with title, description, and completed status.
    """
    description: str
    is_completed: bool
    title: Optional[str] = None
    id: Optional[str] = None
