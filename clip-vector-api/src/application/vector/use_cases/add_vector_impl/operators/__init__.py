"""
RxPY operators for vector processing pipeline.
"""
# Operadores originales (solo los que existen)
from src.application.vector.use_cases.add_vector_impl.operators.save_to_database_operator_rxpy import save_to_database_operator

# Nuevos operadores (estilo generate_vectors.py)
from src.application.vector.use_cases.add_vector_impl.operators.load_image_operator_rxpy import load_image_operator
from src.application.vector.use_cases.add_vector_impl.operators.detect_objects_with_dino_operator_rxpy import detect_objects_with_dino_operator
from src.application.vector.use_cases.add_vector_impl.operators.encode_objects_with_dinov3_operator_rxpy import encode_objects_with_dinov3_operator
from src.application.vector.use_cases.add_vector_impl.operators.combine_detection_and_image_operator_rxpy import combine_detection_and_image_operator
from src.application.vector.use_cases.add_vector_impl.operators.combine_encoding_with_command_operator_rxpy import combine_encoding_with_command_operator
from src.application.vector.use_cases.add_vector_impl.operators.save_vectors_to_db_operator_rxpy import save_vectors_to_db_operator
from src.application.vector.use_cases.add_vector_impl.operators.update_faiss_with_multiple_vectors_operator_rxpy import update_faiss_with_multiple_vectors_operator
from src.application.vector.use_cases.add_vector_impl.operators.to_add_vector_response_operator_rxpy import to_add_vector_response_operator

__all__ = [
    # Originales
    "save_to_database_operator",
    # Nuevos
    "load_image_operator",
    "detect_objects_with_dino_operator",
    "encode_objects_with_dinov3_operator",
    "combine_detection_and_image_operator",
    "combine_encoding_with_command_operator",
    "save_vectors_to_db_operator",
    "update_faiss_with_multiple_vectors_operator",
    "to_add_vector_response_operator",
]

