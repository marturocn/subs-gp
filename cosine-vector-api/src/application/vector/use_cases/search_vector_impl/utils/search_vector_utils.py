"""
Search Vector Utils
"""
from io import BytesIO
from typing import List, Tuple, Union, Dict, Any, Optional

from fastapi import UploadFile

from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_exception import SearchVectorException
from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_format_exception import SearchVectorFormatException


async def transform_search_image_to_binary(image: UploadFile) -> BytesIO:
    """Transform a search image to a binary format."""
    try:
        # Read the image data
        image_data = await image.read()
        image_stream = BytesIO(image_data)
        
        if not image.content_type or not image.content_type.startswith('image/'):
            raise SearchVectorException(detail={"error": "File must be an image"})

        return image_stream
    except SearchVectorException:
        raise
    except Exception as e:
        raise SearchVectorException(detail={"error": str(e)})


def format_search_results(
    search_results: List[Union[Tuple[str, float], Tuple[str, float, Optional[Dict[str, Any]]]]]
) -> List[dict]:
    """
    Format search results into a standardized format.
    
    Args:
        search_results: List of tuples with:
            - (piece_id, similarity_score) - Legacy format
            - (piece_id, combined_score, euclidean_info) - New format with euclidean distance data
        
    Returns:
        List of dictionaries with formatted results:
        {
            "id": str,
            "similarity": float,
            "euclidean_distance": Optional[float],  # Distancia euclidiana entre vectores
            "ranking_method": str  # "cosine_only" o "cosine_euclidean_combined"
        }
        
    Raises:
        SearchVectorFormatException: If formatting fails
    """
    try:
        formatted_results = []
        
        for result in search_results:
            # Determinar el formato de la tupla
            if len(result) == 2:
                # Formato legacy: (piece_id, similarity)
                piece_id, similarity = result
                euclidean_info = None
            elif len(result) == 3:
                # Formato nuevo: (piece_id, combined_score, euclidean_info)
                piece_id, similarity, euclidean_info = result
            else:
                raise ValueError(f"Invalid result tuple length: {len(result)}. Expected 2 or 3 elements.")
            
            # Crear resultado formateado
            formatted_result = {
                "id": piece_id,
                "similarity": float(similarity)
            }
            
            # Agregar información de distancia euclidiana si está disponible
            if euclidean_info is not None and isinstance(euclidean_info, dict):
                formatted_result["euclidean_distance"] = euclidean_info.get('euclidean_distance')
                formatted_result["ranking_method"] = "cosine_euclidean_combined"
            else:
                formatted_result["ranking_method"] = "cosine_only"
            
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        raise SearchVectorFormatException(
            message="Failed to format search results",
            detail={"error": str(e), "result_format": str(type(search_results))}
        )
