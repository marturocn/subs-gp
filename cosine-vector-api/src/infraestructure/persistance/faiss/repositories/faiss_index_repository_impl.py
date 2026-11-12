"""
FAISS Index Repository Implementation - Infrastructure Layer with GPU Support
Uses FAISS-GPU with IndexIDMap2 for efficient vector search with custom IDs
Implements reactive patterns using RxPY/reactivex
"""
import logging
import os
import threading
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

import faiss
from reactivex import Observable, of, throw
from reactivex import operators as ops
from reactivex.scheduler import ImmediateScheduler

from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.infraestructure.framework.config.config_root_module import get_app_config

# Use the global configured logger
try:
    from src.infraestructure import logger
except ImportError:
    # Fallback: configure basic logger if the global one is not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


def _init_cuda_context() -> bool:
    """
    Force CUDA context initialization before checking GPU availability.
    This helps ensure GPU is properly detected when running under uvicorn.
    Returns True if GPU is available.
    """
    try:
        # Force CUDA initialization by setting environment variable
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        
        # Get number of GPUs - this triggers CUDA context init
        num_gpus = faiss.get_num_gpus()
        logger.info(f"🔍 [CUDA] CUDA context check: {num_gpus} GPU(s) detected")
        
        if num_gpus > 0:
            # Try to create resources to validate GPU access
            try:
                test_resources = faiss.StandardGpuResources() # pyright: ignore[reportAttributeAccessIssue]
                del test_resources
                logger.info("✅ [CUDA] GPU resources successfully validated")
                return True
            except Exception as e:
                logger.warning(f"⚠️ [CUDA] GPU detected but resources failed: {e}")
                return False
        return False
    except Exception as e:
        logger.warning(f"⚠️ [CUDA] CUDA init failed: {e}")
        return False


class FaissIndexRepositoryImpl(FaissIndexRepository):
    """
    Implementation of FAISS index service with GPU support.
    Uses IndexIDMap2 for custom ID mapping and GPU acceleration for fast search.
    All methods return Observable for reactive programming patterns.
    Uses lazy initialization to ensure CUDA context is ready.
    Implemented as a singleton to ensure the same index is used across requests.
    """
    
    _instance: Optional['FaissIndexRepositoryImpl'] = None
    _instance_lock = threading.Lock()
    _instance_id: int = 0  # For debugging
    _my_id: int = 0  # Instance identifier for debugging
    
    def __new__(cls) -> 'FaissIndexRepositoryImpl':
        """Ensure only one instance is created (Singleton pattern)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance_id += 1
                    logger.info(f"🆕 [SINGLETON] Creating new FaissIndexRepositoryImpl instance #{cls._instance_id}")
                    cls._instance = super().__new__(cls)
                    cls._instance._is_initialized_singleton = False
                    cls._instance._my_id = cls._instance_id
        else:
            logger.debug("♻️ [SINGLETON] Reusing existing FaissIndexRepositoryImpl instance")
        return cls._instance
    
    def __init__(self):
        # Skip re-initialization if already done
        if hasattr(self, '_is_initialized_singleton') and self._is_initialized_singleton:
            logger.debug("♻️ [SINGLETON] Instance already initialized, skipping __init__")
            return
            
        logger.info("🔥 Initializing FAISS Index Repository with GPU Support")
        
        # Get configuration
        app_config = get_app_config()
        self.vector_dimension = app_config.faiss_vector_dimension
        
        logger.info(f"📐 [INIT] Vector dimension from config: {self.vector_dimension}")
        
        # Thread lock for thread-safe operations
        self._local_lock = threading.Lock()
        
        # Lazy initialization flags
        self._initialized = False
        self.gpu_available = False
        self.gpu_resources: Optional[Any] = None
        self.index: Optional[Any] = None
        self.piece_id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_piece_id: Dict[int, str] = {}
        
        # Perform deferred initialization
        self._ensure_initialized()
        
        final_vector_count = self.index.ntotal if self.index else 0
        logger.info(f"✅ [INIT] FAISS Index Repository initialized. "
                   f"GPU: {self.gpu_available}, Vectors: {final_vector_count}")
        
        # Mark singleton as fully initialized
        self._is_initialized_singleton = True
    
    def _ensure_initialized(self) -> None:
        """Ensure the FAISS index is initialized (lazy initialization)."""
        if self._initialized:
            return
            
        with self._local_lock:
            if self._initialized:
                return
            
            logger.info("🔄 [LAZY INIT] Starting deferred FAISS initialization...")
            
            # Force CUDA context initialization
            self.gpu_available = _init_cuda_context()
            
            if self.gpu_available:
                logger.info(f"🚀 [INIT] GPU available! Found {faiss.get_num_gpus()} GPU(s)")
                self.gpu_resources = faiss.StandardGpuResources() # pyright: ignore[reportAttributeAccessIssue]
            else:
                logger.warning("⚠️ [INIT] No GPU available, falling back to CPU")
                self.gpu_resources = None
            
            # Initialize empty index
            self._initialize_empty_index()
            
            self._initialized = True
            logger.info("✅ [LAZY INIT] FAISS initialization complete")
    
    def _get_index(self) -> Any:
        """Get the initialized index, ensuring initialization has occurred."""
        self._ensure_initialized()
        if self.index is None:
            raise RuntimeError("FAISS index failed to initialize")
        return self.index
    
    def _initialize_empty_index(self) -> None:
        """Initialize an empty FAISS index with IndexIDMap2."""
        try:
            logger.info("🚀 Initializing empty FAISS IndexIDMap2 index")
            
            # Create base index - IndexFlatIP for inner product (cosine similarity with normalized vectors)
            base_index = faiss.IndexFlatIP(self.vector_dimension)
            
            # Wrap with IndexIDMap2 to support custom IDs and vector reconstruction
            cpu_index = faiss.IndexIDMap2(base_index)
            
            # Move to GPU if available
            if self.gpu_available and self.gpu_resources is not None:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)  # type: ignore
                logger.info("🚀 [INIT] Index moved to GPU successfully")
            else:
                self.index = cpu_index
                logger.info("💻 [INIT] Using CPU index")
            
            # Mapping from piece_id (string) to faiss_id (int)
            self.piece_id_to_faiss_id: Dict[str, int] = {}
            # Mapping from faiss_id (int) to piece_id (string)
            self.faiss_id_to_piece_id: Dict[int, str] = {}
            
            logger.info(f"✅ Empty FAISS IndexIDMap2 initialized with dimension {self.vector_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS index: {e}")
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")
    
    def _create_gpu_index(self, cpu_index: Any) -> Any:
        """Create a GPU index from a CPU index."""
        if self.gpu_available and self.gpu_resources is not None:
            try:
                gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)  # pyright: ignore[reportAttributeAccessIssue]
                logger.info("🚀 Index moved to GPU")
                return gpu_index
            except Exception as e:
                logger.warning(f"⚠️ Failed to move index to GPU: {e}, using CPU")
                return cpu_index
        return cpu_index
    
    def _get_cpu_index(self) -> Any:
        """Get CPU version of the current index for operations that require it."""
        if self.gpu_available and self.index is not None:
            try:
                return faiss.index_gpu_to_cpu(self.index)  # pyright: ignore[reportAttributeAccessIssue]
            except Exception as e:
                logger.warning(f"⚠️ Failed to convert GPU index to CPU: {e}")
                return self.index
        return self.index
    
    def _normalize_vector(self, vector: List[float]) -> np.ndarray:
        """Normalize a vector for cosine similarity."""
        arr = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(arr)
        return arr

    # ==================== REACTIVE METHODS ====================

    def add_vector(self, piece_id: str, vector: List[float]) -> Observable[None]:
        """Add a vector to the FAISS index. Returns Observable."""
        def _add_vector_operation() -> Observable[None]:
            try:
                self._add_vector_internal(piece_id, vector)
                return of(None)
            except Exception as e:
                return throw(e)
        
        return _add_vector_operation()
    
    def add_vector_sync(self, piece_id: str, vector: List[float]) -> Observable[None]:
        """Add a vector to the FAISS index (synchronous). Returns Observable."""
        def _add_vector_sync_operation() -> Observable[None]:
            try:
                self._add_vector_internal(piece_id, vector)
                return of(None)
            except Exception as e:
                return throw(e)
        
        return _add_vector_sync_operation()
    
    def _add_vector_internal(self, piece_id: str, vector: List[float]) -> None:
        """Internal method to add a vector (actual implementation)."""
        try:
            with self._local_lock:
                if piece_id in self.piece_id_to_faiss_id:
                    logger.warning(f"⚠️ Vector with piece_id {piece_id} already exists, updating...")
                    self._remove_vector_internal(piece_id)
                
                # Generate new faiss_id
                faiss_id = self._generate_faiss_id()
                
                # Normalize vector for cosine similarity
                vector_array = self._normalize_vector(vector)
                ids = np.array([faiss_id], dtype=np.int64)
                
                # Add to index
                self.index.add_with_ids(vector_array, ids)  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                
                # Update mappings
                self.piece_id_to_faiss_id[piece_id] = faiss_id
                self.faiss_id_to_piece_id[faiss_id] = piece_id
                
                logger.info(f"✅ Added vector for piece_id={piece_id} with faiss_id={faiss_id}")
                
        except Exception as e:
            logger.error(f"❌ Error adding vector: {e}")
            raise
    
    def _generate_faiss_id(self) -> int:
        """Generate a unique FAISS ID."""
        if not self.faiss_id_to_piece_id:
            return 1
        return max(self.faiss_id_to_piece_id.keys()) + 1
    
    def _remove_vector_internal(self, piece_id: str) -> None:
        """Internal method to remove a vector (assumes lock is held)."""
        if piece_id not in self.piece_id_to_faiss_id:
            return
        
        faiss_id = self.piece_id_to_faiss_id[piece_id]
        ids_to_remove = np.array([faiss_id], dtype=np.int64)
        
        # For GPU index, we need to convert to CPU, remove, and convert back
        if self.gpu_available:
            cpu_index = self._get_cpu_index()
            cpu_index.remove_ids(ids_to_remove)
            self.index = self._create_gpu_index(cpu_index)
        else:
            self.index.remove_ids(ids_to_remove) # pyright: ignore[reportOptionalMemberAccess]
        
        # Update mappings
        del self.piece_id_to_faiss_id[piece_id]
        del self.faiss_id_to_piece_id[faiss_id]
        
        logger.info(f"🗑️ Removed vector for piece_id={piece_id}")
    
    def remove_vector(self, piece_id: str) -> Observable[None]:
        """Remove a vector from the FAISS index. Returns Observable."""
        def _remove_vector_operation() -> Observable[None]:
            try:
                with self._local_lock:
                    if piece_id not in self.piece_id_to_faiss_id:
                        logger.warning(f"⚠️ Vector with piece_id {piece_id} not found")
                        return of(None)
                    self._remove_vector_internal(piece_id)
                return of(None)
            except Exception as e:
                return throw(e)
        
        return _remove_vector_operation()
    
    async def update_vector(self, piece_id: str, vector: List[float]) -> Observable[None]:
        """Update a vector in the FAISS index. Returns Observable."""
        def _update_vector_operation() -> Observable[None]:
            try:
                with self._local_lock:
                    if piece_id in self.piece_id_to_faiss_id:
                        self._remove_vector_internal(piece_id)
                
                self._add_vector_internal(piece_id, vector)
                logger.info(f"✅ Updated vector for piece_id={piece_id}")
                return of(None)
            except Exception as e:
                return throw(e)
        
        return _update_vector_operation()
    
    def search_similar(self, query_vector: List[float], k: int = 5) -> Observable[List[Tuple[str, float]]]:
        """Search for similar vectors in the index. Returns Observable."""
        def _search_operation() -> Observable[List[Tuple[str, float]]]:
            try:
                results = self._search_similar_internal(query_vector, k)
                return of(results)
            except Exception as e:
                return throw(e)
        
        return _search_operation()
    
    def search_similar_sync(self, query_vector: List[float], k: int = 5) -> Observable[List[Tuple[str, float]]]:
        """Search for similar vectors in the index (synchronous). Returns Observable."""
        try:
            results = self._search_similar_internal(query_vector, k)
            logger.info(f"🔍 [SEARCH_SYNC] Returning {len(results)} results via Observable")
            # Return Observable that emits immediately
            return of(results)
        except Exception as e:
            logger.error(f"❌ [SEARCH_SYNC] Error: {e}")
            return throw(e)
    
    def _convert_search_results(
        self, distances: np.ndarray, indices: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Convert FAISS search results to piece_id, similarity tuples."""
        results = []
        matched = 0
        unmatched = 0
        
        for dist, idx in zip(distances[0], indices[0]):
            idx_int = int(idx)
            if idx_int == -1:
                continue
            if idx_int in self.faiss_id_to_piece_id:
                piece_id = self.faiss_id_to_piece_id[idx_int]
                results.append((piece_id, float(dist)))
                matched += 1
            else:
                unmatched += 1
                if unmatched <= 3:
                    logger.debug(f"⚠️ [SEARCH] idx_int {idx_int} not in mapping")
        
        if unmatched > 0:
            logger.warning(f"⚠️ [SEARCH] {unmatched} indices not found in mapping, {matched} matched")
        
        return results
    
    def _search_similar_internal(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Internal method to search for similar vectors."""
        try:
            with self._local_lock:
                if self.index is None or self.index.ntotal == 0:
                    logger.info(f"📭 [SEARCH] No vectors in index (Instance #{self._my_id})")
                    return []
                
                query_array = self._normalize_vector(query_vector)
                search_k = min(k, self.index.ntotal)
                distances, indices = self.index.search(query_array, search_k)  # pyright: ignore[reportCallIssue]
                
                results = self._convert_search_results(distances, indices)
                logger.info(f"🔍 [SEARCH] Found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"❌ [SEARCH] Error searching vectors: {e}")
            return []
    
    def get_all_vectors(self) -> Observable[List[Tuple[str, List[float]]]]:
        """Get all vectors from the index. Returns Observable."""
        def _get_all_operation() -> Observable[List[Tuple[str, List[float]]]]:
            try:
                results = self._get_all_vectors_internal()
                return of(results)
            except Exception as e:
                return throw(e)
        
        return _get_all_operation()
    
    def _get_all_vectors_internal(self) -> List[Tuple[str, List[float]]]:
        """
        Internal method to get all vectors from the index.
        
        Returns:
            List of tuples (piece_id, vector)
        """
        try:
            with self._local_lock:
                if self.index is None or self.index.ntotal == 0:
                    logger.info("📭 [GET_ALL] No vectors in index")
                    return []
                
                results = []
                cpu_index = self._get_cpu_index()
                
                for faiss_id, piece_id in self.faiss_id_to_piece_id.items():
                    try:
                        vector = np.zeros(self.vector_dimension, dtype=np.float32)
                        cpu_index.reconstruct(faiss_id, vector)
                        results.append((piece_id, vector.tolist()))
                    except Exception as e:
                        logger.warning(f"⚠️ [GET_ALL] Error reconstructing vector {faiss_id}: {e}")
                
                logger.info(f"✅ [GET_ALL] Retrieved {len(results)} vectors")
                return results
                
        except Exception as e:
            logger.error(f"❌ [GET_ALL] Error getting all vectors: {e}")
            return []
    
    def rebuild_index_from_vectors(self, vectors: List[Tuple[int, List[float]]]) -> Observable[int]:
        """
        Rebuild the entire FAISS index from a list of vectors.
        Returns Observable with the number of vectors added.
        """
        def _rebuild_operation() -> Observable[int]:
            try:
                count = self._rebuild_index_internal(vectors)
                return of(count)
            except Exception as e:
                return throw(e)
        
        return _rebuild_operation()
    
    def _rebuild_index_internal(self, vectors: List[Tuple[int, List[float]]]) -> int:
        """
        Internal method to rebuild the entire FAISS index.
        
        Args:
            vectors: List of tuples (faiss_id, vector) where faiss_id is the FAISS index ID
            
        Returns:
            Number of vectors successfully added to the index
        """
        try:
            with self._local_lock:
                logger.info(f"🔄 [REBUILD] Instance #{self._my_id} - Starting rebuild with {len(vectors)} vectors")
                
                if len(vectors) == 0:
                    logger.info("📭 [REBUILD] No vectors to add, creating empty index")
                    self._initialize_empty_index()
                    return 0
                
                # Detect vector dimension from input
                detected_dimension = self._detect_vector_dimension(vectors)
                if detected_dimension != self.vector_dimension:
                    logger.warning(f"⚠️ [REBUILD] Dimension mismatch: config={self.vector_dimension}, detected={detected_dimension}")
                    self.vector_dimension = detected_dimension
                
                # Create new base index
                base_index = faiss.IndexFlatIP(self.vector_dimension)
                cpu_index = faiss.IndexIDMap2(base_index)
                
                # Prepare vectors and IDs
                valid_vectors = []
                valid_ids = []
                new_piece_id_to_faiss_id = {}
                new_faiss_id_to_piece_id = {}
                
                for faiss_id, vector in vectors:
                    if len(vector) != self.vector_dimension:
                        logger.debug(f"⏭️ [REBUILD] Skipping vector {faiss_id}: dimension {len(vector)} != {self.vector_dimension}")
                        continue
                    
                    valid_vectors.append(vector)
                    valid_ids.append(faiss_id)
                    
                    piece_id = str(faiss_id)
                    new_piece_id_to_faiss_id[piece_id] = faiss_id
                    new_faiss_id_to_piece_id[faiss_id] = piece_id
                
                if len(valid_vectors) == 0:
                    logger.warning("⚠️ [REBUILD] No valid vectors after filtering")
                    self._initialize_empty_index()
                    return 0
                
                # Convert to numpy arrays
                vectors_array = np.array(valid_vectors, dtype=np.float32)
                ids_array = np.array(valid_ids, dtype=np.int64)
                
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(vectors_array)
                
                # Add vectors with IDs
                cpu_index.add_with_ids(vectors_array, ids_array)  # pyright: ignore[reportCallIssue]
                
                logger.info(f"✅ [REBUILD] Added {cpu_index.ntotal} vectors to CPU index")
                
                # Move to GPU if available
                self.index = self._create_gpu_index(cpu_index)
                
                # Update mappings
                self.piece_id_to_faiss_id = new_piece_id_to_faiss_id
                self.faiss_id_to_piece_id = new_faiss_id_to_piece_id
                
                # Debug: verify mappings were set correctly
                logger.info(f"🔍 [REBUILD] Instance #{self._my_id} - faiss_id_to_piece_id size after rebuild: {len(self.faiss_id_to_piece_id)}")
                if len(self.faiss_id_to_piece_id) > 0:
                    sample = list(self.faiss_id_to_piece_id.items())[:3]
                    logger.info(f"🔍 [REBUILD] Instance #{self._my_id} - Sample mapping entries: {sample}")
                
                final_count = self.index.ntotal  # type: ignore
                logger.info(f"✅ [REBUILD] Instance #{self._my_id} - Rebuild completed: {final_count} vectors in index, {len(self.faiss_id_to_piece_id)} mappings")
                
                return final_count
                
        except Exception as e:
            logger.error(f"❌ [REBUILD] Error rebuilding index: {e}")
            import traceback
            logger.error(f"❌ [REBUILD] Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to rebuild FAISS index: {e}")
    
    def _detect_vector_dimension(self, vectors: List[Tuple[int, List[float]]]) -> int:
        """Detect the vector dimension from input vectors."""
        dimension_counts = Counter(len(vector) for _, vector in vectors)
        
        if len(dimension_counts) > 1:
            logger.warning(f"⚠️ [REBUILD] Multiple dimensions found: {dict(dimension_counts)}")
        
        detected_dimension = dimension_counts.most_common(1)[0][0]
        logger.info(f"📏 [REBUILD] Detected dimension: {detected_dimension}")
        
        return detected_dimension
    
    def get_index_status(self) -> Observable[Dict[str, Any]]:
        """Get the current status of the FAISS index. Returns Observable."""
        def _get_status_operation() -> Observable[Dict[str, Any]]:
            try:
                status = self._get_index_status_internal()
                return of(status)
            except Exception as e:
                return throw(e)
        
        return _get_status_operation()
    
    def _get_index_status_internal(self) -> Dict[str, Any]:
        """
        Internal method to get the current status of the FAISS index.
        
        Returns:
            Dictionary with index status information
        """
        try:
            with self._local_lock:
                total_vectors = self.index.ntotal if self.index else 0
                
                return {
                    'total_vectors': total_vectors,
                    'total_mappings': len(self.piece_id_to_faiss_id),
                    'vector_dimension': self.vector_dimension,
                    'gpu_enabled': self.gpu_available,
                    'gpu_count': faiss.get_num_gpus(),
                    'index_type': 'IndexIDMap2(IndexFlatIP)',
                }
        except Exception as e:
            logger.error(f"❌ Error getting index status: {e}")
            return {'error': str(e)}
    
    def get_vector_by_id(self, faiss_id: str) -> Observable[List[float]]:
        """Get a specific vector by its piece ID. Returns Observable."""
        def _get_vector_operation() -> Observable[List[float]]:
            try:
                vector = self._get_vector_by_id_internal(faiss_id)
                return of(vector)
            except Exception as e:
                return throw(e)
        
        return _get_vector_operation()
    
    def _get_vector_by_id_internal(self, faiss_id: str) -> List[float]:
        """
        Internal method to get a specific vector by its piece ID using reconstruct.

        Args:
            faiss_id: The piece ID of the vector (string)

        Returns:
            The vector as a list of floats
        """
        try:
            with self._local_lock:
                if self.index is None:
                    raise ValueError("FAISS index not initialized")

                piece_id = faiss_id

                if piece_id not in self.piece_id_to_faiss_id:
                    raise ValueError(f"piece_id {piece_id} not found in mapping")

                actual_faiss_id = self.piece_id_to_faiss_id[piece_id]
                cpu_index = self._get_cpu_index()

                vector = np.zeros(self.vector_dimension, dtype=np.float32)
                cpu_index.reconstruct(actual_faiss_id, vector)

                return vector.tolist()

        except Exception as e:
            logger.error(f"❌ [GET_VECTOR_BY_ID] Error reconstructing vector {faiss_id}: {e}")
            raise
    
    # ==================== NON-REACTIVE UTILITY METHODS ====================
    
    def clear_vectors(self) -> None:
        """Clear all vectors from the index."""
        try:
            with self._local_lock:
                logger.info("🗑️ [CLEAR] Clearing all vectors from index")
                self._initialize_empty_index()
                logger.info("✅ [CLEAR] Index cleared successfully")
        except Exception as e:
            logger.error(f"❌ [CLEAR] Error clearing index: {e}")
            raise
    
    def diagnose_index(self) -> Dict[str, Any]:
        """Diagnose the state of the index for debugging."""
        try:
            with self._local_lock:
                total_vectors = self.index.ntotal if self.index else 0
                
                diagnosis = {
                    'index': {
                        'total_vectors': total_vectors,
                        'vector_dimension': self.vector_dimension,
                        'index_initialized': self.index is not None,
                        'gpu_enabled': self.gpu_available,
                    },
                    'mappings': {
                        'piece_id_to_faiss_id_count': len(self.piece_id_to_faiss_id),
                        'faiss_id_to_piece_id_count': len(self.faiss_id_to_piece_id),
                        'sample_piece_ids': list(self.piece_id_to_faiss_id.keys())[:5],
                    },
                    'consistency_check': {
                        'mappings_match': len(self.piece_id_to_faiss_id) == len(self.faiss_id_to_piece_id),
                        'vectors_match_mappings': total_vectors == len(self.piece_id_to_faiss_id),
                    }
                }
                
                logger.info(f"🔍 [DIAGNOSE] Diagnosis: {diagnosis}")
                return diagnosis
                
        except Exception as e:
            logger.error(f"❌ [DIAGNOSE] Error in diagnosis: {e}")
            return {'error': str(e)}
