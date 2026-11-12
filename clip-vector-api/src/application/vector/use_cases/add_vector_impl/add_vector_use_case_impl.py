"""
Add Vector Use Case Implementation with Grounding DINO + DINOv3 - Application Layer

Pipeline similar a generate_vectors.py:
1. Cargar imagen original
2. Detectar objetos con GroundingDINO
3. Codificar cada objeto con DINOv3
4. Combinar vectores en uno principal
5. Guardar vectores en BD (principal + objetos)
6. Actualizar índice FAISS
"""
from reactivex import Observable, of
from reactivex.operators import catch, zip as zip_op
from typing import Optional

from src.domain.piece.repositories.piece_vector_repository import PieceVectorRepository
from src.domain.piece.repositories.piece_object_repository import PieceObjectRepository
from src.infraestructure.persistance.model.config.grounding_dino_config import ARTWORK_DINO_ADVANCED_CONFIG
from src.common.exceptions.custom_server_exception import CustomServerException
from src.domain.vector.use_cases.add_vector_case import AddVectorUseCase, AddVectorCommand, AddVectorResponse
from src.domain.model.repositories.dinov3_model_repository import DINOv3ModelRepository
from src.domain.model.repositories.grounding_dino_repository import GroundingDinoRepository
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.application.vector.use_cases.add_vector_impl.exceptions.add_vector_exception import AddVectorException
from src.infraestructure import logger

# Importar nuevos operadores
from src.application.vector.use_cases.add_vector_impl.operators.load_image_operator_rxpy import load_image_operator
from src.application.vector.use_cases.add_vector_impl.operators.detect_objects_with_dino_operator_rxpy import detect_objects_with_dino_operator
from src.application.vector.use_cases.add_vector_impl.operators.combine_detection_and_image_operator_rxpy import combine_detection_and_image_operator
from src.application.vector.use_cases.add_vector_impl.operators.encode_objects_with_dinov3_operator_rxpy import encode_objects_with_dinov3_operator
from src.application.vector.use_cases.add_vector_impl.operators.combine_encoding_with_command_operator_rxpy import combine_encoding_with_command_operator
from src.application.vector.use_cases.add_vector_impl.operators.save_vectors_to_db_operator_rxpy import save_vectors_to_db_operator
from src.application.vector.use_cases.add_vector_impl.operators.update_faiss_with_multiple_vectors_operator_rxpy import update_faiss_with_multiple_vectors_operator
from src.application.vector.use_cases.add_vector_impl.operators.to_add_vector_response_operator_rxpy import to_add_vector_response_operator


class AddVectorUseCaseImpl(AddVectorUseCase):
    """
    Implementation of the Add Vector use case using Grounding DINO + DINOv3.
    
    Flow (similar to generate_vectors.py):
    1. Load original image
    2. Detect objects with Grounding DINO (optimized for art galleries)
    3. Encode each object with DINOv3 (generates multiple vectors)
    4. Combine all vectors into ONE weighted average vector
    5. Save main vector + individual object vectors to database
    6. Update FAISS index with all vectors
    
    Configuration:
    - Uses ARTWORK_DINO_ADVANCED_CONFIG: Optimized for paintings, sculptures, and art pieces
    - DINOv3: Vision transformer for generating image embeddings
    - Weighted combination: Objects with higher confidence have more influence
    """
    
    def __init__(
        self,
        piece_repository: PieceVectorRepository,
        piece_object_repository: PieceObjectRepository,
        dino_model_repository: GroundingDinoRepository,
        dinov3_model_repository: DINOv3ModelRepository,
    ):
        self._piece_repository = piece_repository
        self._piece_object_repository = piece_object_repository
        self._dino_model_repository = dino_model_repository
        self._dinov3_model_repository = dinov3_model_repository
        # Use artwork-specific configuration (no text prompt needed)
        self._text_prompt = ARTWORK_DINO_ADVANCED_CONFIG.text_prompt
    
    def execute(self, command: AddVectorCommand) -> Observable[AddVectorResponse]:
        """
        Execute the add vector use case using RxPY pipeline with Grounding DINO + DINOv3.
        Similar to generate_vectors.py workflow.
        
        Args:
            command: Command with vector addition data
            
        Returns:
            Observable[AddVectorResponse]: Observable that emits the response
        """
        logger.info("="*70)
        logger.info("🎨 INICIANDO PIPELINE DE GENERACIÓN DE VECTORES")
        logger.info(f"📂 piece_id: {command.piece_id}")
        logger.info("="*70)
        
        # Pipeline secuencial similar a generate_vectors.py
        # Crear dos streams paralelos y combinarlos
        image_stream = of(command).pipe(
            load_image_operator(logger)
        )
        
        detection_stream = of(command).pipe(
            detect_objects_with_dino_operator(
                self._dino_model_repository,
                self._text_prompt,
                logger
            )
        )
        
        # Combinar los dos streams
        combined_stream = image_stream.pipe(
            zip_op(detection_stream),
            combine_detection_and_image_operator(logger)
        )
        
        # Continuar con encoding
        encoding_stream = combined_stream.pipe(
            encode_objects_with_dinov3_operator(
                self._dinov3_model_repository,
                self._dino_model_repository,
                logger
            )
        )
        
        # Combinar encoding con command
        final_stream = combined_stream.pipe(
            zip_op(encoding_stream),
            combine_encoding_with_command_operator(logger)
        )
        
        # Continuar con guardado y respuesta
        return final_stream.pipe(
            # Step 6: Guardar vectores en base de datos (principal + objetos) y objetos en piece_object
            save_vectors_to_db_operator(
                self._piece_repository,
                self._piece_object_repository,
                logger
            ),
            # Step 7: Actualizar índice FAISS con todos los vectores
            update_faiss_with_multiple_vectors_operator(
                logger
            ),
            # Step 8: Transformar a AddVectorResponse
            to_add_vector_response_operator(logger),
            # Error handling
            catch(self._handle_error)
        )
    
    def _handle_error(self, error: Exception, source: Observable) -> Observable[AddVectorResponse]:
        """Handle and transform errors appropriately."""
        logger.error(f"❌ Error inesperado en AddVectorUseCaseImpl: {error}")
        logger.error(f"   Tipo de error: {type(error).__name__}")

        if isinstance(error, (CustomServerException)):
            raise error
        
        raise AddVectorException(
            message=f"Error inesperado en AddVectorUseCaseImpl: {str(error)}",
            detail={"error": str(error), "error_type": type(error).__name__}
        )
