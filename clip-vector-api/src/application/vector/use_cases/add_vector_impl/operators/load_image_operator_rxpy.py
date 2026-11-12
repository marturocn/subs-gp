"""
RxPY Operator for loading and preparing the original image.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
from typing import Callable, Dict, Any
from PIL import Image
from reactivex import Observable
from reactivex.operators import map, flat_map
import reactivex as rx

from src.domain.vector.use_cases.add_vector_case.commands.add_vector_command import AddVectorCommand
from src.application.vector.use_cases.add_vector_impl.exceptions.image_to_binary_exception import ImageToBinaryException


class LoadImageOperator:
    """
    Operador para cargar y preparar la imagen original.
    """

    @staticmethod
    def execute(
        command: AddVectorCommand,
        logger: logging.Logger
    ) -> Observable[Dict[str, Any]]:
        """
        Carga la imagen desde BytesIO y la prepara para procesamiento.

        :param command: AddVectorCommand con image_data
        :param logger: Logger instance
        :return: Observable[Dict] con command e imagen PIL
        """
        try:
            logger.info("🖼️  Cargando imagen original...")
            
            # Reset stream position
            command.image_data.seek(0)
            
            # Load image
            original_image = Image.open(command.image_data)
            
            # Convert to RGB if necessary
            if original_image.mode != 'RGB':
                if original_image.mode == 'RGBA':
                    logger.info("   🔄 Convirtiendo RGBA a RGB...")
                    background = Image.new('RGB', original_image.size, (255, 255, 255))
                    background.paste(original_image, mask=original_image.split()[3] if len(original_image.split()) > 3 else None)
                    original_image = background
                else:
                    logger.info(f"   🔄 Convirtiendo {original_image.mode} a RGB...")
                    original_image = original_image.convert('RGB')
            
            logger.info(f"✅ Imagen cargada: {original_image.size[0]}x{original_image.size[1]} píxeles")
            
            return rx.of({
                "command": command,
                "original_image": original_image
            })
            
        except Exception as e:
            logger.error(f"❌ Error cargando imagen: {e}")
            return rx.throw(ImageToBinaryException(
                message="Failed to load image",
                detail={"error": str(e)}
            ))


def load_image_operator(
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para cargar y preparar la imagen original.
    
    :param logger: Logger instance
    :return: Observable con command e imagen PIL
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        return source.pipe(
            flat_map(lambda command: LoadImageOperator.execute(
                command,
                logger
            ))
        )
    return operator
