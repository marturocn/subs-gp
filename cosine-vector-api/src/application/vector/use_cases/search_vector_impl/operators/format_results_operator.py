"""
RxPY Operator for formatting search results.
"""
import logging
from typing import List, Tuple, Callable, Dict, Any, Union, Optional
from reactivex import Observable
from reactivex import operators as ops

from src.domain.vector.use_cases.search_vector_case import SearchVectorResponse
from src.application.vector.use_cases.search_vector_impl.utils.search_vector_utils import format_search_results
from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_format_exception import SearchVectorFormatException

from src.infraestructure import logger

def format_results_operator() -> Callable[[Observable], Observable]:
    
    def format_results_step(search_results: Union[List[Tuple[str, float]], Dict[str, Any]]) -> SearchVectorResponse:
        """Format search results into response."""
        try:
            
                # Cast to proper type for format_search_results
                typed_results: List[Union[Tuple[str, float], Tuple[str, float, Optional[Dict[str, Any]]]]] = search_results  # type: ignore
                formatted_results = format_search_results(typed_results)
                logger.info(f"📦 [FORMAT_RESULTS] Formatted {len(formatted_results)} legacy results")
                
                # Add method field to legacy results
                for result in formatted_results:
                    result['method'] = 'search_vectors_operator'
                
                return SearchVectorResponse(results=formatted_results)
                
        except Exception as e:
            logger.error(f"❌ [FORMAT_RESULTS] Error formatting search results: {e}")
            raise SearchVectorFormatException(detail={"error": str(e)})
    
    return ops.map(format_results_step)
