"""
Response DTOs for Task application layer.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TaskResponseDTO(BaseModel):
    """
    DTO for a single task response via API.
    """
    id: str = Field(..., description="Task unique identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    is_completed: bool = Field(..., description="Task completion status")
    created_at: Optional[datetime] = Field(default=None, description="Task creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Task last update timestamp")

    class Config:
        from_attributes = True


class TaskListResponseDTO(BaseModel):
    """
    DTO for a list of tasks with pagination info via API.
    """
    tasks: List[TaskResponseDTO] = Field(..., description="List of tasks")
    total: int = Field(..., ge=0, description="Total number of tasks")
    limit: Optional[int] = Field(default=None, ge=1, description="Applied limit")
    offset: Optional[int] = Field(default=None, ge=0, description="Applied offset")
    has_more: bool = Field(default=False, description="Whether there are more tasks available")

    class Config:
        from_attributes = True
