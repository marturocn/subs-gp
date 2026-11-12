"""
RxPY Operator for formatting load vectors response.
"""
import logging
from typing import Callable
from reactivex import Observable
from reactivex import operators as ops

from src.domain.vector.use_cases.load_vectors_from_db_case import LoadVectorsFromDbResponse


logger = logging.getLogger(__name__)


def format_load_response_operator() -> Callable[[Observable], Observable]:
    """
    RxPY operator that formats the load vectors response.
    
    Returns:
        RxPY operator function that transforms int (loaded count) to LoadVectorsFromDbResponse
    """
    
    def format_response_step(loaded_count: int) -> LoadVectorsFromDbResponse:
        """Format the final response."""
        logger.info(f"📄 [FORMAT_RESPONSE_OP] Formatting response for {loaded_count} loaded vectors")
        
        return LoadVectorsFromDbResponse(
            message="Vectors successfully loaded from database.",
            loaded_count=loaded_count
        )
    
    return ops.map(format_response_step)
