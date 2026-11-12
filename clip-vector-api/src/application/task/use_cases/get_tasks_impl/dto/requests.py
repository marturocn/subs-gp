"""
Request DTOs for get tasks use case.
"""
from pydantic import BaseModel, Field
from typing import Optional


class GetTasksRequestDTO(BaseModel):
    """
    DTO for getting tasks with filters via API.
    """
    limit: Optional[int] = Field(default=None, ge=1, le=1000, description="Maximum number of tasks to return")
    offset: Optional[int] = Field(default=None, ge=0, description="Number of tasks to skip")
    is_completed: Optional[bool] = Field(default=None, description="Filter by completion status")
    search: Optional[str] = Field(default=None, min_length=1, description="Search term")
