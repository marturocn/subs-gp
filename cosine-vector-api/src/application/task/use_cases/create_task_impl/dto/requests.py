"""
Request DTOs for create task use case.
"""
from pydantic import BaseModel, Field
from src.domain.task import CreateTaskCommand


class CreateTaskRequestDTO(BaseModel):
    """
    DTO for creating a new task via API.
    """
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: str = Field(..., min_length=1, max_length=1000, description="Task description")
    is_completed: bool = Field(default=False, description="Task completion status")
    
    def to_command(self) -> CreateTaskCommand:
        """
        Convert this request to domain command.
        
        Returns:
            CreateTaskCommand: Domain command object
        """
        return CreateTaskCommand(
            title=self.title,
            description=self.description,
            is_completed=self.is_completed
        )
