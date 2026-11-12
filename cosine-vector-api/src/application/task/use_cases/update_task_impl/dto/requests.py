"""
Request DTOs for update task use case.
"""
from pydantic import BaseModel, Field
from typing import Optional
from src.domain.task import UpdateTaskCommand


class UpdateTaskRequestDTO(BaseModel):
    """
    DTO for updating an existing task via API.
    """
    title: Optional[str] = Field(default=None, min_length=1, max_length=200, description="Updated task title")
    description: Optional[str] = Field(default=None, min_length=1, max_length=1000, description="Updated task description")
    is_completed: Optional[bool] = Field(default=None, description="Updated completion status")
    
    def to_command(self, task_id: str) -> UpdateTaskCommand:
        """
        Convert this request to domain command.
        
        Args:
            task_id: The ID of the task to update
            
        Returns:
            UpdateTaskCommand: Domain command object
        """
        return UpdateTaskCommand(
            id=task_id,
            title=self.title,
            description=self.description,
            is_completed=self.is_completed
        )
