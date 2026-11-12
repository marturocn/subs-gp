"""
RxPY Operator for loading image from BytesIO.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
from typing import Callable
from io import BytesIO
from PIL import Image
from reactivex import Observable
from reactivex.operators import map, flat_map, catch
import reactivex as rx

from src.application.vector.use_cases.generate_vector_impl.exceptions.generate_vector_exception import GenerateVectorException


class LoadImageOperator:
    """
    Operador para cargar una imagen desde BytesIO y convertirla a PIL Image.
    """

    @staticmethod
    def execute(
        image_data: BytesIO,
        logger: logging.Logger
    ) -> Observable[Image.Image]:
        """
        Carga la imagen desde BytesIO y la convierte a PIL Image en formato RGB.

        :param image_data: BytesIO con los datos de la imagen
        :param logger: Logger instance
        :return: Observable[Image.Image] con la imagen cargada
        """
        try:
            logger.info("📂 Cargando imagen desde BytesIO...")
            
            # Reset stream position
            image_data.seek(0)
            
            # Cargar imagen
            image = Image.open(image_data)
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                logger.info(f"   🔄 Convirtiendo imagen de {image.mode} a RGB")
                if image.mode == 'RGBA':
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
                    image = background
                else:
                    image = image.convert('RGB')
            
            logger.info(f"✅ Imagen cargada exitosamente: {image.size[0]}x{image.size[1]} píxeles")
            
            return rx.of(image)
            
        except Exception as e:
            logger.error(f"❌ Error cargando imagen: {e}")
            return rx.throw(GenerateVectorException(
                message="Error loading image from BytesIO",
                detail={"error": str(e)}
            ))


def load_image_operator(
    logger: logging.Logger
) -> Callable[[Observable], Observable[Image.Image]]:
    """
    Operator para cargar imagen desde BytesIO.
    
    :param logger: Logger instance
    :return: Observable con PIL Image
    """
    def operator(source: Observable) -> Observable[Image.Image]:
        return source.pipe(
            flat_map(lambda command: LoadImageOperator.execute(
                command.image_data,
                logger
            ))
        )
    return operator
