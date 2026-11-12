"""
Vector Controller - API Layer
"""
from http import HTTPMethod
from fastapi import APIRouter, status
from dishka.integrations.fastapi import inject

from src.application.vector.use_cases.search_vector_impl.utils.search_vector_utils import transform_search_image_to_binary
from src.domain.vector.use_cases.search_vector_case import SearchVectorUseCase, SearchVectorCommand
from src.domain.vector.use_cases.load_vectors_from_db_case import LoadVectorsFromDbUseCase, LoadVectorsFromDbCommand
from common.core.observable_resolver import resolve_observable_to_single_result
from infraestructure.framework.validators.validate_upload_file import ValidateUploadFile
from infraestructure.framework.validators.validate_query_params import ValidateQueryParams
from src.api.vector.dto.search_image_dto import SearchImageDto
from src.api.vector.dto.search_query_params_dto import SearchQueryParamsDto


@inject
class VectorController:
    """
    Controller for vector operations.
    """
    
    # Constants
    APPLICATION_JSON = "application/json"

    def __init__(
        self,
        search_vector_use_case: SearchVectorUseCase,
        load_vectors_from_db_use_case: LoadVectorsFromDbUseCase,
    ):
        self._search_vector_use_case = search_vector_use_case
        self._load_vectors_from_db_use_case = load_vectors_from_db_use_case
        self.router = APIRouter(prefix="/vector", tags=["vector"])
        self._register_routes()

    def _register_routes(self):
        """Register all routes for the vector controller."""
        self.router.add_api_route(
            "/search",
            self.search_vector,
            methods=[f"{HTTPMethod.POST}"],
            description="Buscar imágenes similares en el índice FAISS con filtros opcionales (threshold, maxResults)",
            responses={
                status.HTTP_200_OK: {
                    "description": "Búsqueda realizada exitosamente",
                    "content": {
                        self.APPLICATION_JSON: {
                            "example": {
                                "status": 200,
                                "meta": {},
                                "data": {
                                    "results": [
                                        {"id": "abc123", "similarity": 0.942},
                                        {"id": "xyz456", "similarity": 0.891}
                                    ]
                                }
                            }
                        }
                    }
                },
                status.HTTP_400_BAD_REQUEST: {
                    "description": "Imagen inválida o no se encontraron coincidencias",
                    "content": {
                        self.APPLICATION_JSON: {
                            "examples": {
                                "invalid_image": {
                                    "summary": "Imagen inválida",
                                    "value": {
                                        "status": 400,
                                        "meta": {},
                                        "error": {
                                            "code": 3002,
                                            "message": "Invalid or unreadable image.",
                                            "detail": {"reason": "Invalid or unreadable image"}
                                        }
                                    }
                                },
                                "no_results": {
                                    "summary": "No se encontraron coincidencias",
                                    "value": {
                                        "status": 400,
                                        "meta": {},
                                        "error": {
                                            "code": 3006,
                                            "message": "No similar vectors found.",
                                            "detail": {"reason": "No similar vectors found in the index"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                status.HTTP_500_INTERNAL_SERVER_ERROR: {
                    "description": "Error interno de búsqueda o carga del modelo",
                    "content": {
                        self.APPLICATION_JSON: {
                            "examples": {
                                "search_error": {
                                    "summary": "Error de búsqueda",
                                    "value": {
                                        "status": 500,
                                        "meta": {},
                                        "error": {
                                            "code": 3008,
                                            "message": "Error while searching in index.",
                                            "detail": {"reason": "Internal error with the search"}
                                        }
                                    }
                                },
                                "model_error": {
                                    "summary": "Error del modelo",
                                    "value": {
                                        "status": 500,
                                        "meta": {},
                                        "error": {
                                            "code": 3009,
                                            "message": "Error loading the CLIP model.",
                                            "detail": {"reason": "Internal error with the model"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            status_code=status.HTTP_200_OK,
        )

        # Register load vectors from database route
        self.router.add_api_route(
            "/load_from_db",
            self.load_vectors_from_db,
            methods=[f"{HTTPMethod.GET}"],
            description="Cargar vectores desde la base de datos al índice FAISS",
            responses={
                status.HTTP_200_OK: {
                    "description": "Vectores cargados exitosamente",
                    "content": {
                        self.APPLICATION_JSON: {
                            "example": {
                                "status": 200,
                                "meta": {},
                                "data": {
                                    "message": "Vectors successfully loaded from database.",
                                    "loaded_count": 237
                                }
                            }
                        }
                    }
                },
                status.HTTP_500_INTERNAL_SERVER_ERROR: {
                    "description": "Error interno durante la carga de vectores",
                    "content": {
                        self.APPLICATION_JSON: {
                            "examples": {
                                "database_error": {
                                    "summary": "Error de base de datos",
                                    "value": {
                                        "status": 500,
                                        "meta": {},
                                        "error": {
                                            "code": 1005,
                                            "message": "Error loading vectors from database.",
                                            "detail": {"reason": "Error querying the DB"}
                                        }
                                    }
                                },
                                "invalid_format": {
                                    "summary": "Formato de vector inválido",
                                    "value": {
                                        "status": 500,
                                        "meta": {},
                                        "error": {
                                            "code": 3004,
                                            "message": "One or more vectors have invalid format or dimension.",
                                            "detail": {"reason": "One or more vectors with invalid shape", "invalid_ids": [12, 45]}
                                        }
                                    }
                                },
                                "faiss_error": {
                                    "summary": "Error del índice FAISS",
                                    "value": {
                                        "status": 500,
                                        "meta": {},
                                        "error": {
                                            "code": 1008,
                                            "message": "Error updating FAISS index.",
                                            "detail": {"reason": "General FAISS index failure", "faiss_error": "..."}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            status_code=status.HTTP_200_OK,
        )

    async def search_vector(
        self,
        search_dto: SearchImageDto = ValidateUploadFile(
            SearchImageDto, 
            description="Imagen a buscar en el índice FAISS"
        )(),
        query_params: SearchQueryParamsDto = ValidateQueryParams(
            SearchQueryParamsDto,
            description="Parámetros de búsqueda opcionales"
        )()
    ):
        """
        Busca imágenes similares en el índice FAISS.
        
        Este endpoint:
        1. Recibe una imagen
        2. Convierte la imagen en un vector usando el modelo CLIP
        3. Busca vectores similares en el índice FAISS
        4. Devuelve los resultados ordenados por similitud
        
        Args:
            search_dto: DTO con el archivo de imagen (JPG, PNG, etc.)
            query_params: DTO con parámetros de búsqueda (threshold, maxResults)
            
        Returns:
            JSONResponse con los resultados de la búsqueda
        """
        from infraestructure import logger
        
        try:
            logger.info("🎯 VectorController: Iniciando search_vector")
            logger.info(f"📊 Query params: threshold={query_params.threshold}, maxResults={query_params.maxResults}, enableObjectDetector={query_params.enableObjectDetector}")

            # Read image data
            logger.info("📷 VectorController: Procesando imagen...")
            image_stream = await transform_search_image_to_binary(search_dto.file)
            
            # Get size safely for BytesIO
            try:
                stream_size = image_stream.seek(0, 2)  # Seek to end to get size
                image_stream.seek(0)  # Reset position
                logger.info(f"📷 VectorController: Imagen procesada - Size: {stream_size} bytes")
            except (IOError, OSError):
                logger.info("📷 VectorController: Imagen procesada - Size: unknown")
            
            # Create command with parameters including enableObjectDetector
            command = SearchVectorCommand(
                image_data=image_stream,
                threshold=query_params.threshold,
                max_results=query_params.maxResults,
                enable_object_detector=query_params.enableObjectDetector
            )
            logger.info(f"⚡ VectorController: Comando creado con threshold={query_params.threshold}, maxResults={query_params.maxResults}, enableObjectDetector={query_params.enableObjectDetector}")
            
            # Execute use case
            response_observable = self._search_vector_use_case.execute(command)
            logger.info("🔄 VectorController: Observable creado, resolviendo resultado...")
            
            result = await resolve_observable_to_single_result(response_observable)
            logger.info(f"✅ VectorController: Resultado obtenido - Type: {type(result)}")
            
            # Log result details
            if hasattr(result, 'status'):
                logger.info(f"📊 VectorController: Result status: {getattr(result, 'status', 'unknown')}")
            
            # Format response
            logger.info("📤 VectorController: Enviando respuesta al cliente")
            return result
            
        except Exception as e:
            logger.error(f"❌ VectorController: Error en search_vector: {type(e).__name__}: {str(e)}")
            raise

    async def load_vectors_from_db(self):
        """
        Carga vectores desde la base de datos al índice FAISS.
        
        Este endpoint:
        1. Obtiene todos los vectores almacenados en la base de datos
        2. Valida el formato y dimensión de los vectores
        3. Reconstruye completamente el índice FAISS
        4. Devuelve el número de vectores cargados
        
        Returns:
            JSONResponse con el resultado de la operación de carga
        """
        # Create command (no parameters needed)
        command = LoadVectorsFromDbCommand()
        
        # Execute use case
        response_observable = self._load_vectors_from_db_use_case.execute(command)
        result = await resolve_observable_to_single_result(response_observable)
        
        # Format response
        return result
