"""
Task list response model.
"""
from dataclasses import dataclass
from typing import List, Optional
from .task_response import TaskResponse


@dataclass
class GetTasksResponse:
    """
    Response model for a list of tasks with pagination info.
    """
    tasks: List[TaskResponse]
    total: int
    limit: Optional[int] = None
    offset: Optional[int] = None
    has_more: bool = False
