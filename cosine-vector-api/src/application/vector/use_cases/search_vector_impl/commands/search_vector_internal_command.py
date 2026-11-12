"""
Internal commands for search vector implementation.
These are application-layer specific commands that may differ from domain commands.
"""
from dataclasses import dataclass
from typing import List, Optional
from io import BytesIO


@dataclass
class SearchVectorInternalCommand:
    """
    Internal command for search vector operations at application layer.
    This may contain additional processing data or transformed inputs.
    """
    image_data: BytesIO
    k: Optional[int] = 5
    # Additional fields specific to application layer processing can be added here


@dataclass
class SearchVectorInternalResponse:
    """
    Internal response for search vector operations at application layer.
    This may contain additional processing data or intermediate results.
    """
    results: List[dict]
    # Additional fields specific to application layer processing can be added here
