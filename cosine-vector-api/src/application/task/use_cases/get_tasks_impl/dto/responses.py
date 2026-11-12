"""
Response DTOs for get tasks use case.
"""
from dataclasses import dataclass
from typing import List
from src.domain.task.models.entities.task import Task


@dataclass
class GetTasksResponse:
    """
    Response DTO for getting tasks.
    """
    tasks: List[Task]
    total: int
