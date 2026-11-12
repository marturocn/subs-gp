"""
RxPY Operator for combining detection results with original image.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
from typing import Callable, Dict, Any
from reactivex import Observable
from reactivex.operators import map
import reactivex as rx


def combine_detection_and_image_operator(
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para combinar resultados de detección con imagen original.

    :param logger: Logger instance
    :return: Observable con detection_result y original_image combinados
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        def combine(data_tuple):
            """
            Combina los datos de detección con la imagen original.
            data_tuple = (image_data, detection_result)
            """
            image_data, detection_result = data_tuple
            
            logger.info("🔗 Combinando imagen con resultados de detección...")
            
            return {
                "command": image_data["command"],
                "original_image": image_data["original_image"],
                "detection_result": detection_result
            }
        
        return source.pipe(
            map(combine)
        )
    return operator
