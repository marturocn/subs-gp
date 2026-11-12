"""
Response DTOs for update task use case.
"""
from dataclasses import dataclass
from src.domain.task.models.entities.task import Task


@dataclass
class UpdateTaskResponse:
    """
    Response DTO for updating a task.
    """
    task: Task
    message: str = "Task updated successfully"
