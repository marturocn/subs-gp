"""
Model Controller - API Layer
"""
from http import HTTPMethod
from fastapi import APIRouter, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from dishka.integrations.fastapi import inject
from io import BytesIO
from typing import Optional

from api.model.dto import DetectObjectsDto
from api.vector.dto.add_image_dto import AddImageDto
from infraestructure.framework.validators.validate_upload_file import ValidateUploadFile
from src.domain.model.use_cases.detect_objects_case import DetectObjectsUseCase, DetectObjectsCommand
from src.application.vector.use_cases.add_vector_impl.utils.transform_image_to_binary import transform_image_to_binary
from common.core.observable_resolver import resolve_observable_to_single_result


@inject
class ModelController:
    """
    Controller for AI model operations (object detection, etc.).
    """

    def __init__(
        self,
        detect_objects_use_case: DetectObjectsUseCase,
    ):
        self._detect_objects_use_case = detect_objects_use_case
        self.router = APIRouter(prefix="/model", tags=["model"])
        self._register_routes()

    def _register_routes(self):
        """Register all routes for the model controller."""
        self.router.add_api_route(
            "/items",
            self.detect_objects,
            methods=[f"{HTTPMethod.POST}"],
            description="Detectar objetos en una imagen usando Grounding DINO (Google Lens-like)",
            responses={
                status.HTTP_200_OK: {
                    "description": "Objetos detectados exitosamente",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": 200,
                                "meta": {},
                                "data": {
                                    "detections": {
                                        "objects": [
                                            {
                                                "label": "artwork",
                                                "confidence": 0.8542,
                                                "bounding_box": {
                                                    "x1": 100,
                                                    "y1": 150,
                                                    "x2": 500,
                                                    "y2": 600,
                                                    "width": 400,
                                                    "height": 450
                                                },
                                                "center": {
                                                    "x": 300,
                                                    "y": 375
                                                },
                                                "area": 180000
                                            }
                                        ],
                                        "count": 1,
                                        "text_prompt": "artwork . painting . sculpture",
                                        "image_dimensions": {
                                            "width": 1920,
                                            "height": 1080
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                status.HTTP_400_BAD_REQUEST: {
                    "description": "Imagen inválida o ilegible"
                },
                status.HTTP_500_INTERNAL_SERVER_ERROR: {
                    "description": "Error interno del servidor"
                }
            },
            status_code=status.HTTP_200_OK,
        )

    async def detect_objects(
        self,
        file: AddImageDto = ValidateUploadFile(
            AddImageDto, 
            description="Imagen para detectar objetos"
        )(),
        text_prompt: Optional[str] = Form(
            None,
            description="Prompt de texto opcional para detección (ej: 'person . car . dog'). Si no se proporciona, usa el valor por defecto.",
            example="artwork . painting . sculpture . piece"
        )
    ):
        """
        Detecta objetos en una imagen usando Grounding DINO.
        
        Este endpoint proporciona funcionalidad similar a Google Lens:
        1. Recibe una imagen y un prompt opcional
        2. Detecta objetos usando Grounding DINO con el prompt especificado
        3. Devuelve los objetos detectados con sus coordenadas de bounding box
        
        Args:
            file: Archivo de imagen (JPG, PNG, etc.)
            text_prompt: Prompt de texto opcional para guiar la detección.
                        Formato: "objeto1 . objeto2 . objeto3"
                        Si no se proporciona, usa el valor por defecto de las variables de entorno.
            
        Returns:
            JSONResponse con los objetos detectados y sus coordenadas
        """
        # Read image data
        image_stream = await transform_image_to_binary(file.file)
        
        # Create command
        command = DetectObjectsCommand(
            image_data=image_stream,
            text_prompt=text_prompt
        )
        
        # Execute use case
        response_observable = self._detect_objects_use_case.execute(command)
        result = await resolve_observable_to_single_result(response_observable)
        
        # Return response
        return result
