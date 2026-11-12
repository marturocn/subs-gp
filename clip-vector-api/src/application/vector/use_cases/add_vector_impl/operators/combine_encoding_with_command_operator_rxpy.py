"""
RxPY Operator for combining encoding results with command.
Follows the pattern of save_to_database_operator_rxpy.py
"""
import logging
from typing import Callable, Dict, Any
from reactivex import Observable
from reactivex.operators import map


def combine_encoding_with_command_operator(
    logger: logging.Logger
) -> Callable[[Observable], Observable[Dict[str, Any]]]:
    """
    Operator para combinar resultados de encoding con command.

    :param logger: Logger instance
    :return: Observable con encoding_result y command combinados
    """
    def operator(source: Observable) -> Observable[Dict[str, Any]]:
        def combine(data_tuple):
            """
            Combina los datos de encoding con el command.
            data_tuple = (previous_data, encoding_result)
            """
            previous_data, encoding_result = data_tuple
            
            logger.info("🔗 Combinando resultados de encoding con command...")
            
            return {
                "command": previous_data["command"],
                "encoding_result": encoding_result
            }
        
        return source.pipe(
            map(combine)
        )
    return operator
