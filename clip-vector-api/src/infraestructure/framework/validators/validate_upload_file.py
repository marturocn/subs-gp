from fastapi import Depends, UploadFile, File, HTTPException
from pydantic import BaseModel, ValidationError
from typing import Type, Any, Optional, Set
import io
from PIL import Image
# Add HEIF support if pillow-heif is installed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    # If pillow-heif is not installed, continue without HEIF support
    pass

from src.common.exceptions.http.generic_validation_exception import GenericValidationException

# Constantes
INVALID_FILE_MESSAGE = "Invalid file"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg", 
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/heif",
    "image/heic"
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heif", ".heic"}


class ValidateUploadFile:
    """
    Validador para archivos subidos, especialmente imágenes.
    
    Realiza validaciones de:
    - Tipo de contenido
    - Tamaño del archivo
    - Formato de imagen válido
    - Extensión del archivo
    """
    
    def __init__(
        self, 
        dto_class: Type[BaseModel],
        description: Optional[str] = None,
        max_size: Optional[int] = None,
        allowed_types: Optional[Set[str]] = None,
        validate_image: bool = True
    ):
        self.dto_class = dto_class
        self.description = description or "Archivo a validar"
        self.max_size = max_size or MAX_FILE_SIZE
        self.allowed_types = allowed_types or ALLOWED_CONTENT_TYPES
        self.validate_image = validate_image

    def _validate_file_exists(self, file: Optional[UploadFile]) -> None:
        """Validar que el archivo existe y tiene nombre."""
        if not file or not file.filename:
            raise GenericValidationException(
                message="No file provided",
                detail={"field": "file", "reason": "File is required"}
            )

    def _validate_content_type(self, file: UploadFile) -> None:
        """Validar tipo de contenido del archivo."""
        if file.content_type in self.allowed_types:
            return

        # Special case for HEIC/HEIF files
        file_extension = file.filename.lower().split('.')[-1] if file.filename else ''
        if file_extension in ['heic', 'heif'] and file.content_type == 'application/octet-stream':
            return

        raise GenericValidationException(
            message="Invalid file type",
            detail={
                "field": "file",
                "reason": f"File type {file.content_type} not allowed. Allowed types: {', '.join(self.allowed_types)}"
            }
        )

    def _validate_extension(self, file: UploadFile) -> None:
        """Validar extensión del archivo."""
        if file.filename:
            file_extension = file.filename.lower().split('.')[-1]
            if f".{file_extension}" not in ALLOWED_EXTENSIONS:
                raise GenericValidationException(
                    message="Invalid file extension",
                    detail={
                        "field": "file",
                        "reason": f"Extension .{file_extension} not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
                    }
                )

    def _validate_file_size(self, file_size: int) -> None:
        """Validar tamaño del archivo."""
        if file_size > self.max_size:
            raise GenericValidationException(
                message="File too large",
                detail={
                    "field": "file",
                    "reason": f"File size {file_size} bytes exceeds maximum allowed size of {self.max_size} bytes"
                }
            )

        if file_size == 0:
            raise GenericValidationException(
                message="Empty file",
                detail={"field": "file", "reason": "File cannot be empty"}
            )

    def __call__(self) -> Any:
        async def dependency(file: Optional[UploadFile] = File(None, description=self.description)):
            try:
                # Validaciones básicas
                self._validate_file_exists(file)
                self._validate_content_type(file) # pyright: ignore[reportArgumentType]
                self._validate_extension(file) # pyright: ignore[reportArgumentType]

                # Leer el contenido del archivo para validaciones adicionales
                file_content = await file.read() # pyright: ignore[reportOptionalMemberAccess]
                file_size = len(file_content)

                # Validar tamaño
                self._validate_file_size(file_size)

                # Validar formato de imagen si está habilitado
                if self.validate_image:
                    self._validate_image_format(file_content)

                # Restablecer el archivo para que pueda ser leído después
                await file.seek(0) # pyright: ignore[reportOptionalMemberAccess]

                # Crear el DTO usando model_validate
                return self.dto_class.model_validate({"file": file})

            except GenericValidationException:
                # Re-lanzar excepciones de validación personalizadas
                raise
            except Exception as exc:
                # Manejar otros errores inesperados
                raise GenericValidationException(
                    message="File validation error",
                    detail={
                        "field": "file",
                        "reason": f"Unexpected error during file validation: {str(exc)}"
                    }
                )

        return Depends(dependency)
    
    async def _validate_file_content(self, file: UploadFile) -> tuple[bytes, int]:
        """Read and validate file content size."""
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > self.max_size:
            raise GenericValidationException(
                message="File too large",
                detail={
                    "field": "file",
                    "reason": f"File size {file_size} bytes exceeds maximum allowed size of {self.max_size} bytes"
                }
            )
        
        if file_size == 0:
            raise GenericValidationException(
                message="Empty file",
                detail={"field": "file", "reason": "File cannot be empty"}
            )
        
        return file_content, file_size

    async def validate_direct(self, file: Optional[UploadFile]) -> Any:
        """
        Método para validación directa sin Depends (útil para testing).
        """
        try:
            # Validar que el archivo no esté vacío
            if not file or not file.filename:
                raise GenericValidationException(
                    message="No file provided",
                    detail={"field": "file", "reason": "File is required"}
                )

            # Validar tipo de contenido
            if file.content_type not in self.allowed_types:
                # Special case for HEIC/HEIF files that might have wrong content-type
                file_extension = file.filename.lower().split('.')[-1] if file.filename else ''
                if not (file_extension in ['heic', 'heif'] and file.content_type == 'application/octet-stream'):
                    raise GenericValidationException(
                        message="Invalid file type",
                        detail={
                            "field": "file", 
                            "reason": f"File type {file.content_type} not allowed. Allowed types: {', '.join(self.allowed_types)}"
                        }
                    )

            # Validar extensión del archivo
            if file.filename:
                file_extension = file.filename.lower().split('.')[-1]
                if f".{file_extension}" not in ALLOWED_EXTENSIONS:
                    raise GenericValidationException(
                        message="Invalid file extension",
                        detail={
                            "field": "file",
                            "reason": f"Extension .{file_extension} not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
                        }
                    )

            # Leer y validar contenido del archivo
            file_content, _ = await self._validate_file_content(file)

            # Validar formato de imagen si está habilitado
            if self.validate_image:
                self._validate_image_format(file_content)

            # Restablecer el archivo para que pueda ser leído después
            await file.seek(0)

            # Crear el DTO usando model_validate
            return self.dto_class.model_validate({"file": file})

        except GenericValidationException:
            # Re-lanzar excepciones de validación personalizadas
            raise
        except Exception as exc:
            # Manejar otros errores inesperados
            raise GenericValidationException(
                message="File validation error",
                detail={
                    "field": "file",
                    "reason": f"Unexpected error during file validation: {str(exc)}"
                }
            )

    def _validate_image_format(self, file_content: bytes) -> None:
        """
        Valida que el contenido del archivo sea una imagen válida.
        
        Args:
            file_content: Contenido del archivo en bytes
            
        Raises:
            GenericValidationException: Si la imagen no es válida
        """
        try:
            # Intentar abrir la imagen con PIL
            image = Image.open(io.BytesIO(file_content))
            
            # Verificar que la imagen sea válida
            image.verify()
            
            # Validar dimensiones mínimas (opcional)
            image = Image.open(io.BytesIO(file_content))  # Reabrir después de verify()
            width, height = image.size
            
            if width < 10 or height < 10:
                raise GenericValidationException(
                    message="Image too small",
                    detail={
                        "field": "file",
                        "reason": f"Image dimensions {width}x{height} are too small. Minimum size is 10x10 pixels"
                    }
                )
                
        except GenericValidationException:
            # Re-lanzar excepciones de validación personalizadas
            raise
        except Exception as exc:
            # La imagen no es válida o no se puede abrir
            raise GenericValidationException(
                message="Invalid image format",
                detail={
                    "field": "file",
                    "reason": f"File is not a valid image or is corrupted: {str(exc)}"
                }
            )
