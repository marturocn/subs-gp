"""
Vector Controller - API Layer
"""
from http import HTTPMethod
from fastapi import APIRouter, UploadFile, File, Path, status
from fastapi.responses import JSONResponse
from dishka.integrations.fastapi import inject
from inspect import getdoc
from io import BytesIO

from api.vector.dto.add_image_dto import AddImageDto
from api.vector.dto.piece_id_dto import PieceIdDto
from infraestructure.framework.validators.validate_upload_file import ValidateUploadFile
from src.domain.vector.use_cases.add_vector_case import AddVectorUseCase, AddVectorCommand
from src.domain.vector.use_cases.generate_vector_case import GenerateVectorUseCase, GenerateVectorCommand
from src.application.vector.use_cases.add_vector_impl.utils.transform_image_to_binary import transform_image_to_binary
from common.core.observable_resolver import resolve_observable_to_single_result
from infraestructure.framework.validators.validate_params import ValidateParams
from src.common.exceptions.custom_server_exception import CustomServerException
from src.common.exceptions.codes_exception import VectorErrorCodes
from src.common.exceptions.messages_exception import VectorErrorMessages

# Content type constant
JSON_CONTENT_TYPE = "application/json"


@inject
class VectorController:
    """
    Controller for vector operations.
    """

    def __init__(
        self,
        add_vector_use_case: AddVectorUseCase,
        generate_vector_use_case: GenerateVectorUseCase,
    ):
        self._add_vector_use_case = add_vector_use_case
        self._generate_vector_use_case = generate_vector_use_case
        self.router = APIRouter(prefix="/vector", tags=["vector"])
        self._register_routes()

    def _register_routes(self):
        """Register all routes for the vector controller."""
        self.router.add_api_route(
            "/add/{id}",
            self.add_vector,
            methods=[f"{HTTPMethod.POST}"],
            description="Agregar vector CLIP para una pieza artística",
            responses={
                status.HTTP_200_OK: {
                    "description": "Vector agregado exitosamente",
                },
                status.HTTP_400_BAD_REQUEST: {"description": "ID inválido o imagen ilegible"},
                status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Error interno del servidor"},
            },
            status_code=status.HTTP_200_OK,
        )
        
        self.router.add_api_route(
            "",
            self.generate_vector,
            methods=[f"{HTTPMethod.POST}"],
            description="Generar vector CLIP a partir de una imagen",
            responses={
                status.HTTP_200_OK: {
                    "description": "Vector generado exitosamente",
                    "content": {
                        JSON_CONTENT_TYPE: {
                            "example": {
                                "status": 200,
                                "meta": {},
                                "data": {
                                    "message": "Vector generated successfully.",
                                    "vector": [0.123, 0.456, 0.789]
                                }
                            }
                        }
                    }
                },
                status.HTTP_400_BAD_REQUEST: {
                    "description": "Imagen inválida o ilegible",
                    "content": {
                        JSON_CONTENT_TYPE: {
                            "example": {
                                "status": 400,
                                "meta": {},
                                "error": {
                                    "code": 3002,
                                    "message": "Invalid or unreadable image.",
                                    "detail": {"reason": "Invalid or unreadable image"}
                                }
                            }
                        }
                    }
                },
                status.HTTP_500_INTERNAL_SERVER_ERROR: {
                    "description": "Error al cargar el modelo",
                    "content": {
                        JSON_CONTENT_TYPE: {
                            "example": {
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
            },
            status_code=status.HTTP_200_OK,
        )

    async def add_vector(
        self,
        id: PieceIdDto = ValidateParams(PieceIdDto)(),
        file: AddImageDto = ValidateUploadFile(AddImageDto, description="Imagen a añadir al índice FAISS")()
    ):
        """
        Agrega un vector CLIP para una pieza artística.
        
        Este endpoint:
        1. Recibe un ID de pieza y una imagen
        2. Convierte la imagen en un vector usando el modelo CLIP
        3. Guarda el vector en la base de datos
        4. Añade el vector directamente al índice FAISS en memoria para búsquedas inmediatas
        
        Args:
            id: Identificador único de la pieza artística
            file: Archivo de imagen (JPG, PNG, etc.)
            
        Returns:
            JSONResponse con el resultado de la operación
        """
            # # Validate file type
            # if not file.content_type or not file.content_type.startswith('image/'):
            #     raise CustomServerException(
            #         code=VectorErrorCodes.INVALID_IMAGE,
            #         message=VectorErrorMessages.INVALID_IMAGE,
            #         status_code=400
            #     ).set_detail({"reason": "File must be an image"})
            
            # Read image data
        image_stream = await transform_image_to_binary(file.file)
            # Create command
        command = AddVectorCommand(
            piece_id=str(id.id),
            image_data=image_stream
        )
            
            # Execute use case
        response_observable = self._add_vector_use_case.execute(command)
        result = await resolve_observable_to_single_result(response_observable)
            
            # Format response
        return result
            

    async def generate_vector(
        self,
        file: AddImageDto = ValidateUploadFile(
            AddImageDto, 
            description="Imagen a procesar con CLIP"
        )(),
        enable_object_detector: bool = True
    ):
        """
        Genera un vector CLIP a partir de una imagen.
        
        Este endpoint:
        1. Recibe una imagen
        2. Convierte la imagen en un vector usando el modelo CLIP
        3. Devuelve el vector directamente sin guardarlo
        
        Args:
            file: Archivo de imagen (JPG, PNG, etc.)
            enable_object_detector: Si es True, detecta objetos en la imagen. Si es False, solo procesa la imagen completa.
            
        Returns:
            JSONResponse con el vector generado
        """
            # Validate file type

            # Read image data
        image_stream = await transform_image_to_binary(file.file)
            
            # Create command
        command = GenerateVectorCommand(
            image_data=image_stream,
            enable_object_detector=enable_object_detector
        )
            
            # Execute use case
        response_observable = self._generate_vector_use_case.execute(command)
        result = await resolve_observable_to_single_result(response_observable)
            
            # Format response
        return result
            
