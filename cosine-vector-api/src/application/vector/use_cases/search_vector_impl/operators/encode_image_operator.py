"""
RxPY Operator for processing image to vector.
"""
import json
import logging
import os
import asyncio
import concurrent.futures
from typing import List, Callable, Tuple, Optional, Dict, Any
from io import BytesIO
import httpx
import reactivex
from reactivex import Observable, from_, defer, throw
from reactivex import operators as ops

from src.domain.vector.use_cases.search_vector_case import SearchVectorCommand
from src.application.vector.use_cases.search_vector_impl.exceptions.search_vector_encode_exception import SearchVectorEncodeException


logger = logging.getLogger(__name__)


def _get_image_to_vector_credentials() -> Optional[Tuple[str, str]]:
    """
    Get authentication credentials from environment variables.
    
    Returns:
        Tuple of (username, password) if credentials are found, None otherwise
    """
    try:
        credentials_env = os.getenv('IMAGE_TO_VECTOR_CREDENTIALS')
        if not credentials_env:
            logger.warning("⚠️ [ENCODE_IMAGE] IMAGE_TO_VECTOR_CREDENTIALS not found in environment")
            return None
            
        credentials_list = json.loads(credentials_env)
        if not credentials_list or len(credentials_list) == 0:
            logger.warning("⚠️ [ENCODE_IMAGE] No credentials found in IMAGE_TO_VECTOR_CREDENTIALS")
            return None
            
        # Use the first credential from the list
        first_credential = credentials_list[0]
        username = first_credential.get('username')
        password = first_credential.get('password')
        
        if not username or not password:
            logger.error("❌ [ENCODE_IMAGE] Invalid credential format in IMAGE_TO_VECTOR_CREDENTIALS")
            return None
            
        logger.info(f"✅ [ENCODE_IMAGE] Successfully loaded credentials for user: {username}")
        return (username, password)
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ [ENCODE_IMAGE] Failed to parse IMAGE_TO_VECTOR_CREDENTIALS: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [ENCODE_IMAGE] Error loading credentials: {e}")
        return None


def _validate_image_data(image_data: BytesIO) -> bool:
    """Check if image data exists and has content."""
    if not image_data:
        logger.error("❌ [ENCODE_IMAGE] No image data provided")
        return False
    
    try:
        image_size = image_data.getbuffer().nbytes
        if image_size == 0:
            logger.error("❌ [ENCODE_IMAGE] Image data is empty")
            return False
    except Exception:
        pass
    
    return True


def _extract_vectors_from_response(response_json: Dict[str, Any]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Extract piece vector and object vectors from API response."""
    vectors_data = response_json.get("data", {}).get("vectors", {})
    piece_vector = vectors_data.get("piece", [])
    object_vectors = vectors_data.get("objects", [])
    
    logger.info(f"📦 [ENCODE_IMAGE] Received piece_vector: dim={len(piece_vector)}")
    logger.info(f"📦 [ENCODE_IMAGE] Received {len(object_vectors)} object vectors")
    
    for i, obj in enumerate(object_vectors):
        obj_type = obj.get('type', 'unknown')
        obj_vector = obj.get('vector', [])
        logger.info(f"   Object {i+1}: type={obj_type}, dim={len(obj_vector)}")
    
    return piece_vector, object_vectors


def _validate_piece_vector_result(piece_vector: List[float]) -> None:
    """Validate that piece vector is not empty."""
    if not piece_vector or len(piece_vector) == 0:
        logger.error("❌ [ENCODE_IMAGE] Piece vector is empty")
        raise SearchVectorEncodeException(
            message="Failed to generate piece vector from image",
            detail={"reason": "Piece vector is empty"}
        )


class EncodeImageOperator:
    """
    Operator class to encode image to vector using external service.
    Follows the same pattern as TypeScript Observable operators.
    Uses RxPY's defer and conditional logic for reactive validation.
    Now handles multiple vectors (piece + detected objects).
    """
    
    @staticmethod
    def execute(command: SearchVectorCommand) -> Observable[Dict[str, Any]]:
        """
        Executes the image encoding operation with reactive validation.
        
        Args:
            command: SearchVectorCommand containing the image data
            
        Returns:
            Observable that emits a dict with structure:
            {
                'piece_vector': List[float],
                'object_vectors': List[Dict[str, Any]]  # [{'type': str, 'vector': List[float]}]
            }
        """
        return defer(lambda _: EncodeImageOperator._validate_and_encode(command)).pipe(
            ops.do_action(
                on_next=lambda result: logger.info(
                    f"✅ [ENCODE_IMAGE] Successfully encoded image: "
                    f"piece_vector dim={len(result.get('piece_vector', []))}, "
                    f"object_vectors={len(result.get('object_vectors', []))}"
                ),
                on_error=lambda e: logger.error(
                    f"❌ [ENCODE_IMAGE] Error encoding image: {type(e).__name__}: {e}"
                )
            )
        )
    
    @staticmethod
    def _validate_and_encode(command: SearchVectorCommand) -> Observable[Dict[str, Any]]:
        """
        Validates credentials and image data before encoding.
        Uses if_then pattern for conditional validation.
        
        Args:
            command: SearchVectorCommand containing the image data
            
        Returns:
            Observable that emits dict with piece_vector and object_vectors or error
        """
        # Validate credentials exist
        credentials = _get_image_to_vector_credentials()
        if not credentials:
            logger.error("❌ [ENCODE_IMAGE] No valid credentials available")
            return throw(SearchVectorEncodeException(
                message="Authentication credentials not configured",
                detail={"reason": "IMAGE_TO_VECTOR_CREDENTIALS not found or invalid"}
            ))
        
        # Validate image data
        if not _validate_image_data(command.image_data):
            return throw(SearchVectorEncodeException(
                message="Image data is empty",
                detail={"reason": "image_data is empty or None"}
            ))
        
        # If all validations pass, proceed with encoding
        logger.info("✅ [ENCODE_IMAGE] Validations passed, proceeding with encoding")
        return EncodeImageOperator._encode_image_async(command, credentials)
    
    @staticmethod
    def _run_async_task(encode_task: Callable) -> Dict[str, Any]:
        """Run async task in appropriate event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, use thread executor
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(encode_task())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_new_loop)
                    return future.result()
            else:
                return asyncio.run(encode_task())
        except Exception as e:
            logger.error(f"❌ [ENCODE_IMAGE] Error in async wrapper: {type(e).__name__}: {e}")
            raise
    
    @staticmethod
    def _encode_image_async(
        command: SearchVectorCommand, 
        credentials: Tuple[str, str]
    ) -> Observable[Dict[str, Any]]:
        """
        Private method to perform async image encoding.
        
        Args:
            command: SearchVectorCommand containing the image data and enableObjectDetector flag
            credentials: Tuple of (username, password) for authentication
            
        Returns:
            Observable that emits dict with piece_vector and object_vectors
        """
        async def encode_task() -> Dict[str, Any]:
            try:
                image_data = command.image_data
                username, password = credentials
                image_to_vector_url = os.getenv('IMAGE_TO_VECTOR__URL', '')
                enable_object_detector = command.enable_object_detector if command.enable_object_detector is not None else True
                
                logger.info(f"🔄 [ENCODE_IMAGE] Sending image to vector service: {image_to_vector_url}")
                logger.info(f"🎯 [ENCODE_IMAGE] enableObjectDetector: {enable_object_detector}")
                
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                        response = await client.post(
                            image_to_vector_url,
                            files={"file": ("img.png", image_data, "image/png")},
                            params={"enableObjectDetector": enable_object_detector},
                            auth=(username, password),
                            headers={"accept": "application/json"}
                        )
                        
                        response.raise_for_status()
                        response_json = response.json()
                        piece_vector, object_vectors = _extract_vectors_from_response(response_json)
                        
                except httpx.HTTPError as e:
                    logger.error(f"❌ [ENCODE_IMAGE] HTTP request failed: {type(e).__name__}: {e}")
                    raise SearchVectorEncodeException(
                        message="Failed to connect to vector encoding service",
                        detail={"error": str(e)}
                    )

                # Validate piece vector
                _validate_piece_vector_result(piece_vector)
                
                # Return structure with multiple vectors
                return {
                    "piece_vector": piece_vector,
                    "object_vectors": object_vectors
                }

            except SearchVectorEncodeException:
                raise
            except Exception as e:
                logger.error(f"❌ [ENCODE_IMAGE] Unexpected error: {type(e).__name__}: {e}")
                raise SearchVectorEncodeException(
                    message="Invalid or unreadable image",
                    detail={"reason": str(e)}
                )
        
        # Run async task and convert to Observable
        result = EncodeImageOperator._run_async_task(encode_task)
        return from_([result])


def encode_image_operator() -> Callable[[Observable], Observable]:
    """
    RxPY operator that encodes image from SearchVectorCommand to multiple vectors.
    
    This operator transforms an Observable<SearchVectorCommand> to Observable<Dict[str, Any]>
    where the dict contains:
    - 'piece_vector': List[float] - Vector for the complete piece
    - 'object_vectors': List[Dict] - List of detected objects with their vectors
    
    Usage:
        observable.pipe(
            encode_image_operator()
        )
        
    Returns:
        RxPY operator function that transforms SearchVectorCommand to Dict with multiple vectors
    """
    return ops.flat_map(
        lambda command: EncodeImageOperator.execute(command)
    )
