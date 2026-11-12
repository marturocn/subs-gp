"""
Update task command for application layer.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class UpdateTaskCommand:
    """
    Command to update an existing task.
    """
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    is_completed: Optional[bool] = None
