"""
FAISS Index Repository Implementation - Infrastructure Layer with Shared Memory
"""
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import reactivex
from reactivex import Observable, of, throw

from src.infraestructure.framework.providers.shared.shared_memory import SharedMemoryFaissManager
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository

# Constants for error messages
FAISS_INDEX_NOT_INITIALIZED_ERROR = "FAISS index not initialized"

# Use the global configured logger
try:
    from src.infraestructure import logger
except ImportError:
    # Fallback: configure basic logger if the global one is not available
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


class FaissIndexRepositoryImpl(FaissIndexRepository):
    """
    Implementation of FAISS index service with shared memory support.
    Keeps the index in RAM using OS shared memory for access from multiple processes.
    """
    
    def __init__(self, shared_memory_name: Optional[str] = None):
        logger.info("🔥 Initializing FAISS Index Repository with Shared Memory")
        
        # Configuration
        self.vector_dimension = 1024  # CLIP ViT-L-14 dimension
        
        # Shared memory manager
        self.shared_memory_manager = SharedMemoryFaissManager()

        logger.info(f"🧠 [INIT] Using shared memory: '{self.shared_memory_manager.shared_memory_name}'")
        
        
        # Local state (cache)
        self.index: Optional[Any] = None
        self.id_mapping: Dict[int, str] = {}
        self.piece_id_mapping: Dict[str, int] = {}
        self._local_lock = threading.Lock()
        
        # Try to load from shared memory
        self._load_from_shared_memory()
        
        # If no data, initialize new index
        if self.index is None:
            logger.info("🆕 [INIT] No data found in shared memory, initializing new index")
            self._initialize_index()
        else:
            vector_count = self.index.ntotal if self.index else 0
            logger.info(f"🔄 [INIT] Loaded from shared memory: {vector_count} vectors")
        
        final_vector_count = self.index.ntotal if self.index else 0
        logger.info(f"✅ [INIT] FAISS Index Repository initialized with shared memory '{shared_memory_name}'. "
                   f"Available vectors: {final_vector_count}")
    
    def _initialize_index(self):
        """Initialize a new FAISS index in memory with IndexIDMap support."""
        try:
            import faiss
            
            logger.info("🚀 Initializing new FAISS index with IndexIDMap in shared memory")

            # Create a flat IP index for exact search
            base_index = faiss.IndexFlatIP(self.vector_dimension)
            
            # Wrap it with IndexIDMap to support custom IDs
            self.index = faiss.IndexIDMap(base_index)
            self.id_mapping = {}
            self.piece_id_mapping = {}
            
            # Save to shared memory
            self._save_to_shared_memory()
            
            logger.info(f"✅ New FAISS IndexIDMap initialized successfully with dimension {self.vector_dimension}")
            
        except ImportError as e:
            logger.error(f"❌ FAISS not available: {e}")
            raise RuntimeError(f"FAISS is required but not installed: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS index: {e}")
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")
    
    def _load_from_shared_memory(self) -> bool:
        """Load the index from shared memory."""
        try:
            result = self.shared_memory_manager.read_from_shared_memory()
            if result is None:
                logger.info("📭 No data found in shared memory")
                return False
            
            self.index, self.id_mapping, self.piece_id_mapping = result
            vector_count = self.index.ntotal if self.index else 0
            logger.info(f"📤 Index loaded from shared memory: {vector_count} vectors, "
                       f"{len(self.id_mapping)} mappings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading from shared memory: {e}")
            return False
    
    def _save_to_shared_memory(self) -> None:
        """Save the current index to shared memory."""
        try:
            if self.index is not None:
                current_vectors = self.index.ntotal
                current_mappings = len(self.id_mapping)
                logger.info(f"💾 [SAVE] Saving to shared memory: {current_vectors} vectors, {current_mappings} mappings")
                
                self.shared_memory_manager.write_to_shared_memory(
                    self.index, self.id_mapping, self.piece_id_mapping
                )
                logger.info("✅ [SAVE] Index saved to shared memory successfully")
            else:
                logger.warning("⚠️ [SAVE] Cannot save: index is None")
            
        except Exception as e:
            logger.error(f"❌ [SAVE] Error saving to shared memory: {e}")
            raise
    
    def _smart_refresh_from_shared_memory(self) -> None:
        """
        Intelligently refresh from shared memory only if necessary.
        Compares timestamps to determine if there is newer data.
        NOTE: This method assumes that _local_lock is already acquired by the caller.
        """
        try:
            result = self.shared_memory_manager.read_from_shared_memory()
            if result is None:
                logger.info("📭 [SMART_REFRESH] No data in shared memory for refresh")
                return
            
            index, id_mapping, piece_id_mapping = result
            shared_vectors = index.ntotal if index else 0
            shared_mappings = len(id_mapping)
            
            logger.info(f"🔍 [SMART_REFRESH] Shared memory data: {shared_vectors} vectors, {shared_mappings} mappings")
            
            # If we don't have a local index, use the one from shared memory
            if self.index is None:
                self.index, self.id_mapping, self.piece_id_mapping = index, id_mapping, piece_id_mapping
                logger.info(f"🔄 [SMART_REFRESH] Index loaded from shared memory: {shared_vectors} vectors")
                if shared_vectors == 0:
                    logger.warning("⚠️ [SMART_REFRESH] Shared memory contains an empty index")
                return
            
            # Compare with local index
            current_vectors = self.index.ntotal if self.index else 0
            current_mappings = len(self.id_mapping)
            
            logger.info(f"🔍 [SMART_REFRESH] Local state: {current_vectors} vectors, {current_mappings} mappings")
            
            # Be more aggressive: update if shared memory has different count OR different mappings
            if shared_vectors != current_vectors or shared_mappings != current_mappings:
                self.index, self.id_mapping, self.piece_id_mapping = index, id_mapping, piece_id_mapping
                logger.info(f"🔄 [SMART_REFRESH] Updated from shared memory: {current_vectors} → {shared_vectors} vectors, {current_mappings} → {shared_mappings} mappings")
            elif shared_vectors == current_vectors == 0:
                logger.info("📭 [SMART_REFRESH] Both indexes are empty (local and shared memory)")
            else:
                logger.info(f"ℹ️ [SMART_REFRESH] Local index is up to date: {current_vectors} vectors, {current_mappings} mappings")
            
        except Exception as e:
            logger.error(f"❌ [SMART_REFRESH] Error in smart refresh from shared memory: {e}")
    
    def _refresh_from_shared_memory(self) -> None:
        """
        Refresh local state from shared memory.
        NOTE: This method assumes that _local_lock is already acquired by the caller.
        """
        try:
            result = self.shared_memory_manager.read_from_shared_memory()
            if result is not None:
                self.index, self.id_mapping, self.piece_id_mapping = result
                vector_count = self.index.ntotal if self.index else 0
                logger.info(f"🔄 State refreshed from shared memory: {vector_count} vectors")
            
        except Exception as e:
            logger.error(f"❌ Error refreshing from shared memory: {e}")
    
    async def add_vector(self, piece_id: str, vector: List[float]) -> None:
        """Add a vector to the index."""
        try:
            self.add_vector_sync(piece_id, vector)
        except Exception as e:
            logger.error(f"❌ Error adding vector: {e}")
            raise RuntimeError(f"Failed to add vector: {e}")
    
    def add_vector_sync(self, piece_id: str, vector: List[float]) -> None:
        """Add a vector to the index (synchronous) with shared memory persistence."""
        logger.info(f"🎯 [ADD_VECTOR] Adding vector for piece_id={piece_id} (dimension={len(vector)})")
        
        with self._local_lock:
            try:
                if self.index is None:
                    raise RuntimeError("FAISS index not initialized")
                
                # Do smart refresh to sync with changes from other processes
                logger.info("🧠 [ADD_VECTOR] Doing smart refresh before adding")
                self._smart_refresh_from_shared_memory()
                
                # Add vector
                self._add_vector_real(piece_id, vector)
                
                # Force save changes to shared memory with aggressive sync
                logger.info("💾 [ADD_VECTOR] Force saving to shared memory")
                self._save_to_shared_memory()
                
                # AGGRESSIVE INVALIDATION: Force other processes to refresh
                logger.info("🔄 [ADD_VECTOR] Forcing aggressive invalidation for other processes")
                self._force_invalidation_for_other_processes()
                
                # Verify the save was successful
                logger.info("🔍 [ADD_VECTOR] Verifying save by reading back from shared memory")
                verification_result = self.shared_memory_manager.read_from_shared_memory()
                if verification_result:
                    verified_index, _, _ = verification_result
                    verified_count = verified_index.ntotal if verified_index else 0
                    logger.info(f"✅ [ADD_VECTOR] Verification: {verified_count} vectors in shared memory")
                else:
                    logger.warning("⚠️ [ADD_VECTOR] Could not verify save - no data read back")
                
                logger.info(f"🎉 [ADD_VECTOR] Vector added successfully for piece_id={piece_id}")
                
            except Exception as e:
                logger.error(f"❌ [ADD_VECTOR] Error adding vector for piece_id={piece_id}: {e}")
                raise RuntimeError(f"Failed to add vector: {e}")
    
    def add_vector_with_id_sync(self, faiss_id: int, vector: List[float]) -> None:
        """
        Add a vector to the index with a specific FAISS ID (synchronous) using IndexIDMap.
        
        Args:
            faiss_id: The specific FAISS ID to assign to this vector
            vector: The vector to add
        """
        logger.info(f"🎯 [ADD_WITH_ID] Adding vector with specific faiss_id={faiss_id} (dimension={len(vector)})")
        
        with self._local_lock:
            try:
                if self.index is None:
                    raise RuntimeError(FAISS_INDEX_NOT_INITIALIZED_ERROR)
                
                # Do smart refresh to sync with changes from other processes
                logger.info("🧠 [ADD_WITH_ID] Doing smart refresh before adding")
                self._smart_refresh_from_shared_memory()
                
                # Add vector with specific ID using IndexIDMap
                self._add_vector_with_id_real(faiss_id, vector)
                
                # Force save changes to shared memory
                logger.info("💾 [ADD_WITH_ID] Force saving to shared memory")
                self._save_to_shared_memory()
                
                # Force invalidation for other processes
                logger.info("🔄 [ADD_WITH_ID] Forcing aggressive invalidation for other processes")
                self._force_invalidation_for_other_processes()
                
                # Verify the save was successful
                logger.info("🔍 [ADD_WITH_ID] Verifying save by reading back from shared memory")
                verification_result = self.shared_memory_manager.read_from_shared_memory()
                if verification_result:
                    verified_index, _, _ = verification_result
                    verified_count = verified_index.ntotal if verified_index else 0
                    logger.info(f"✅ [ADD_WITH_ID] Verification: {verified_count} vectors in shared memory")
                else:
                    logger.warning("⚠️ [ADD_WITH_ID] Could not verify save - no data read back")
                
                logger.info(f"🎉 [ADD_WITH_ID] Vector added successfully with faiss_id={faiss_id}")
                
            except Exception as e:
                logger.error(f"❌ [ADD_WITH_ID] Error adding vector with faiss_id={faiss_id}: {e}")
                raise RuntimeError(f"Failed to add vector with ID: {e}")
    
    def _force_invalidation_for_other_processes(self) -> None:
        """Force invalidation by recreating shared memory block completely."""
        try:
            logger.info("🔥 [INVALIDATION] Starting aggressive invalidation")
            
            # Step 1: Save current state
            current_vectors = self.index.ntotal if self.index else 0
            logger.info(f"� [INVALIDATION] Current state: {current_vectors} vectors")
            
            # Step 2: Force complete recreation of shared memory
            logger.info("💀 [INVALIDATION] Forcing complete shared memory recreation")
            self.shared_memory_manager.force_cleanup()
            
            # Step 3: Wait a moment to ensure other processes detect the change
            import time
            time.sleep(0.2)
            
            # Step 4: Recreate and save data
            logger.info("🆕 [INVALIDATION] Recreating shared memory with new data")
            self.shared_memory_manager = SharedMemoryFaissManager()
            self._save_to_shared_memory()
            
            # Step 5: Verify the new state
            verification_result = self.shared_memory_manager.read_from_shared_memory()
            if verification_result:
                verified_index, _, _ = verification_result
                verified_count = verified_index.ntotal if verified_index else 0
                logger.info(f"✅ [INVALIDATION] Recreation verified: {verified_count} vectors")
            else:
                logger.warning("⚠️ [INVALIDATION] Could not verify recreation")
            
            logger.info("🎉 [INVALIDATION] Aggressive invalidation completed")
            
        except Exception as e:
            logger.error(f"❌ [INVALIDATION] Error in aggressive invalidation: {e}")
            # Don't raise, as this is not critical
    
    def _add_vector_real(self, piece_id: str, vector: List[float]) -> None:
        """Add vector to real FAISS index."""
        logger.info(f"� [REAL_ADD] Adding real vector for piece_id={piece_id}")
        try:
            import numpy as np
            # Verify index is available
            if self.index is None:
                raise ValueError(FAISS_INDEX_NOT_INITIALIZED_ERROR)
            logger.info(f"� [REAL_ADD] Vector dimension: {len(vector)}, expected: {self.vector_dimension}")
            # Convert to numpy array
            vector_array = np.array([vector], dtype=np.float32)
            logger.info(f"📊 [REAL_ADD] NumPy array created: shape={vector_array.shape}, dtype={vector_array.dtype}")
            # Check if piece already exists in index
            if piece_id in self.piece_id_mapping:
                logger.info(f"🔄 [REAL_ADD] Piece {piece_id} already exists, removing previous vector")
                self._remove_vector_real(piece_id)
            # Add new vector
            current_index = self.index.ntotal
            logger.info(f"📍 [REAL_ADD] Adding vector at index: {current_index}")
            self.index.add(vector_array)
            # Update mappings
            self.id_mapping[current_index] = piece_id
            self.piece_id_mapping[piece_id] = current_index
            logger.info(f"✅ [REAL_ADD] Vector added successfully for piece {piece_id} at index {current_index}")
            logger.info(f"📊 [REAL_ADD] Updated mappings: id_mapping_size={len(self.id_mapping)}, "
                        f"piece_mapping_size={len(self.piece_id_mapping)}")
        except Exception as e:
            logger.error(f"❌ [REAL_ADD] Error in FAISS add real: {e}")
            raise
    
    def _add_vector_with_id_real(self, faiss_id: int, vector: List[float]) -> None:
        """
        Add vector to real FAISS IndexIDMap with specific ID.
        
        Args:
            faiss_id: The specific FAISS ID to assign
            vector: The vector to add
        """
        logger.info(f"🎯 [REAL_ADD_ID] Adding vector with faiss_id={faiss_id}")
        try:
            import numpy as np
            
            # Verify index is available
            if self.index is None:
                raise ValueError(FAISS_INDEX_NOT_INITIALIZED_ERROR)
            
            logger.info(f"📊 [REAL_ADD_ID] Vector dimension: {len(vector)}, expected: {self.vector_dimension}")
            
            # Convert to numpy array
            vector_array = np.array([vector], dtype=np.float32)
            logger.info(f"📊 [REAL_ADD_ID] NumPy array created: shape={vector_array.shape}, dtype={vector_array.dtype}")
            
            # Create ID array for IndexIDMap
            id_array = np.array([faiss_id], dtype=np.int64)
            
            # Add vector with specific ID using IndexIDMap
            logger.info(f"📍 [REAL_ADD_ID] Adding vector with ID to IndexIDMap: faiss_id={faiss_id}")
            self.index.add_with_ids(vector_array, id_array)
            
            # Note: With IndexIDMap, we don't need to maintain our own mappings
            # The index itself handles the ID mapping
            
            logger.info(f"✅ [REAL_ADD_ID] Vector added successfully with faiss_id={faiss_id}")
            logger.info(f"📊 [REAL_ADD_ID] Total vectors in index: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"❌ [REAL_ADD_ID] Error in FAISS add with ID: {e}")
            raise
    
    async def remove_vector(self, piece_id: str) -> None:
        """Remove a vector from the index."""
        try:
            with self._local_lock:
                # Use smart refresh to get the latest state
                self._smart_refresh_from_shared_memory()
                
                # Remove vector
                self._remove_vector_real(piece_id)
                
                # Save changes to shared memory
                self._save_to_shared_memory()
                
        except Exception as e:
            logger.error(f"❌ Error removing vector: {e}")
            raise RuntimeError(f"Failed to remove vector: {e}")
    
    def _remove_vector_real(self, piece_id: str) -> None:
        """Remove a vector from the real FAISS index by rebuilding."""
        if piece_id not in self.piece_id_mapping or self.index is None:
            return
        
        try:
            import numpy as np
            import faiss
            
            # Get all vectors except the one to remove
            all_vectors = []
            new_id_mapping = {}
            new_piece_id_mapping = {}
            
            # Reconstruct all vectors except the one to remove
            for i in range(self.index.ntotal):
                if i in self.id_mapping and self.id_mapping[i] != piece_id:
                    # Get vector at index i
                    vector = np.zeros(self.vector_dimension, dtype=np.float32)
                    self.index.reconstruct(i, vector)
                    all_vectors.append(vector)
                    
                    # Update mappings
                    new_index = len(all_vectors) - 1
                    new_id_mapping[new_index] = self.id_mapping[i]
                    new_piece_id_mapping[self.id_mapping[i]] = new_index
            
            # Create new index
            self.index = faiss.IndexFlatL2(self.vector_dimension)
            self.id_mapping = new_id_mapping
            self.piece_id_mapping = new_piece_id_mapping
            
            # Add all vectors back
            if all_vectors:
                vectors_array = np.array(all_vectors, dtype=np.float32)
                self.index.add(vectors_array)
            
            logger.info(f"Removed vector for piece {piece_id}")
            
        except Exception as e:
            logger.error(f"Error removing vector from FAISS index: {e}")
            raise
    
    async def update_vector(self, piece_id: str, vector: List[float]) -> None:
        """Update a vector in the index (same as add_vector)."""
        await self.add_vector(piece_id, vector)
    
    async def search_similar(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors in the index."""
        try:
            with self._local_lock:
                # Use smart refresh to get the latest data if available
                self._smart_refresh_from_shared_memory()
                
                return self._search_real(query_vector, k)
                
        except Exception as e:
            logger.error(f"❌ Error searching vectors: {e}")
            return []
    def search_similar_sync(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in the index (synchronous).
        
        Returns:
            List of tuples (piece_id, similarity_score)
        """
        try:
            return self._search_real(query_vector, k)
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def _search_real(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """Search using real FAISS index."""
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            import numpy as np
            
            # Convert query vector to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Search in FAISS index
            search_k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_array, search_k)
            
            # Convert results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.id_mapping and idx != -1:
                    piece_id = self.id_mapping[idx]
                    # Convert L2 distance to similarity score (0-1)
                    similarity = 1.0 / (1.0 + float(dist))
                    results.append((piece_id, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in FAISS index: {e}")
            return []
    
    def get_shared_memory_info(self) -> Dict[str, Any]:
        """Get information about the state of shared memory."""
        try:
            total_vectors = self.index.ntotal if self.index else 0
            return {
                'shared_memory_name': self.shared_memory_manager.shared_memory_name,
                'total_vectors': total_vectors,
                'total_mappings': len(self.id_mapping),
                'vector_dimension': self.vector_dimension,
                'memory_size_bytes': self.shared_memory_manager.shared_memory.size if self.shared_memory_manager.shared_memory else 0
            }
        except Exception as e:
            logger.error(f"Error getting shared memory info: {e}")
            return {}

    def get_all_vectors(self) -> List[Tuple[str, List[float]]]:
        """
        Get all vectors from the index with shared memory support.
        
        Returns:
            List of tuples (piece_id, vector)
        """
        try:
            with self._local_lock:
                # Log initial state
                initial_vectors = self.index.ntotal if self.index else 0
                logger.info(f"🔍 [GET_ALL] Initial state: {initial_vectors} local vectors")
                
                # Use smart refresh to get the latest data if available
                self._smart_refresh_from_shared_memory()
                
                # Log state after refresh
                post_refresh_vectors = self.index.ntotal if self.index else 0
                logger.info(f"🔄 [GET_ALL] After refresh: {post_refresh_vectors} vectors")
                
                return self._get_all_vectors_real()
                
        except Exception as e:
            logger.error(f"❌ [GET_ALL] Error getting all vectors: {e}")
            return []

    def _get_all_vectors_real(self) -> List[Tuple[str, List[float]]]:
        """Get all vectors from real FAISS index."""
        try:
            if self.index is None:
                logger.warning("⚠️ [GET_ALL_REAL] FAISS index is None")
                return []
                
            total_vectors = self.index.ntotal
            mappings_count = len(self.id_mapping)
            
            logger.info(f"🔍 [GET_ALL_REAL] Index state: total={total_vectors}, mappings={mappings_count}")
            
            if total_vectors == 0:
                logger.info("📭 [GET_ALL_REAL] No vectors in FAISS index")
                if mappings_count > 0:
                    logger.warning(f"⚠️ [GET_ALL_REAL] Inconsistency: {mappings_count} mappings but 0 vectors in FAISS")
                return []
            
            import numpy as np
            
            logger.info(f"📊 [GET_ALL_REAL] Getting all vectors from index: {total_vectors} vectors")
            
            results = []
            missing_mappings = []
            
            # Iterate through all vectors in the index
            for i in range(total_vectors):
                if i in self.id_mapping:
                    piece_id = self.id_mapping[i]
                    try:
                        # Reconstruct vector at index i
                        vector = np.zeros(self.vector_dimension, dtype=np.float32)
                        self.index.reconstruct(i, vector)
                        results.append((piece_id, vector.tolist()))
                    except Exception as e:
                        logger.error(f"❌ [GET_ALL_REAL] Error reconstructing vector {i}: {e}")
                else:
                    missing_mappings.append(i)
            
            if missing_mappings:
                logger.warning(f"⚠️ [GET_ALL_REAL] Vectors without mapping: {missing_mappings}")
            
            logger.info(f"✅ [GET_ALL_REAL] Retrieved {len(results)} vectors from index")
            return results
            
        except Exception as e:
            logger.error(f"❌ [GET_ALL_REAL] Error getting all vectors from FAISS index: {e}")
            return []
        
    def force_refresh(self) -> None:
        """Force an update from shared memory."""
        logger.info("🔄 Forcing refresh from shared memory")
        with self._local_lock:
            self._refresh_from_shared_memory()
    
    def smart_refresh(self) -> None:
        """Do a smart refresh from shared memory."""
        logger.info("🧠 Smart refresh from shared memory")
        with self._local_lock:
            self._smart_refresh_from_shared_memory()
    
    def diagnose_shared_memory(self) -> Dict[str, Any]:
        """Diagnose the state of shared memory for debugging."""
        try:
            with self._local_lock:
                # Local state
                local_vectors = self.index.ntotal if self.index else 0
                local_mappings = len(self.id_mapping)
                local_piece_mappings = len(self.piece_id_mapping)
                # Shared memory state
                result = self.shared_memory_manager.read_from_shared_memory()
                if result is None:
                    shared_info = {
                        'vectors': 0,
                        'id_mappings': 0,
                        'piece_mappings': 0,
                        'exists': False
                    }
                else:
                    index, id_mapping, piece_id_mapping = result
                    shared_info = {
                        'vectors': index.ntotal if index else 0,
                        'id_mappings': len(id_mapping),
                        'piece_mappings': len(piece_id_mapping),
                        'exists': True
                    }
                diagnosis = {
                    'local': {
                        'vectors': local_vectors,
                        'id_mappings': local_mappings,
                        'piece_mappings': local_piece_mappings,
                        'index_initialized': self.index is not None
                    },
                    'shared_memory': shared_info,
                    'memory_info': self.get_shared_memory_info(),
                    'consistency_check': {
                        'local_mappings_match': local_mappings == local_piece_mappings,
                        'shared_mappings_match': shared_info['id_mappings'] == shared_info['piece_mappings'] if shared_info['exists'] else True,
                        'local_vs_shared_vectors': local_vectors == shared_info['vectors'],
                        'local_vs_shared_mappings': local_mappings == shared_info['id_mappings']
                    }
                }
                logger.info(f"🔍 [DIAGNOSE] Full diagnosis: {diagnosis}")
                return diagnosis
        except Exception as e:
            logger.error(f"❌ [DIAGNOSE] Error in diagnosis: {e}")
            return {'error': str(e)}
    
    def reset_shared_memory(self) -> bool:
        """Reset shared memory by forcing a reinitialization."""
        try:
            logger.info("🔄 [RESET] Resetting shared memory")
            with self._local_lock:
                # Force full cleanup
                self.shared_memory_manager.force_cleanup()
                
                # Reinitialize
                self._initialize_index()
                
                logger.info("✅ [RESET] Shared memory reset successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ [RESET] Error resetting shared memory: {e}")
            return False
    
    @staticmethod
    def cleanup_all_shared_memory() -> None:
        """Clean up all known shared memories from the system."""
        # Use a safer approach: only attempt cleanup on known shared memory names
        # Avoid directly listing /dev/shm which is a publicly writable directory
        # We only clean up the specifically known shared memory segments
        known_names = [
            "cosine_vector_api_faiss_index",  # Current default name
            "faiss_index_shared",  # Previous default name
        ]
        
        for name in known_names:
            try:
                import multiprocessing.shared_memory
                shm = multiprocessing.shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
                logger.info(f"🗑️ [CLEANUP_ALL] Deleted shared memory: {name}")
            except FileNotFoundError:
                logger.info(f"📭 [CLEANUP_ALL] Shared memory not found: {name}")
            except Exception as e:
                logger.error(f"❌ [CLEANUP_ALL] Error cleaning {name}: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("🧹 Cleaning up FAISS repository resources")
        self.shared_memory_manager.cleanup()
    
    def __del__(self):
        """Destructor for automatic cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor
