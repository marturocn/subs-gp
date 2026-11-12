"""
Get tasks command for application layer.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GetTasksCommand:
    """
    Command to get tasks.
    This command can be extended in the future to support filtering, pagination, etc.
    """
    # For now, no parameters needed, but can be extended
    limit: Optional[int] = None
    offset: Optional[int] = None
