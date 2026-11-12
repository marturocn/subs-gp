
import logging
import multiprocessing
import multiprocessing.shared_memory
import pickle
import struct
import threading
import time
from typing import Dict, Optional, Tuple, Any
from src.infraestructure.framework.config.config_root_module import get_app_config
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


class SharedMemoryFaissManager:
    """
    Shared memory manager for FAISS indexes.
    Handles serialization, synchronization, and concurrent access.
    """
    
    def __init__(self, shared_memory_name: Optional[str] = None, 
         initial_size_mb: Optional[int] = None):
        # Obtener la configuración de FAISS
        app_config = get_app_config()
        
        self.shared_memory_name = shared_memory_name or app_config.faiss_shared_memory_name
        self.initial_size_bytes = (initial_size_mb or app_config.faiss_shared_memory_size) * 1024 * 1024
        self.shared_memory: Optional[multiprocessing.shared_memory.SharedMemory] = None
        self.lock = multiprocessing.Lock()
        self._local_lock = threading.Lock()
        # Memory block structure:
        # [4 bytes: data size] [serialized data]
        self.header_size = 4
        logger.info(f"🧠 Initializing SharedMemoryFaissManager with name='{self.shared_memory_name}' size={(initial_size_mb or app_config.faiss_shared_memory_size)}MB")
    
    def _get_or_create_shared_memory(self, required_size: Optional[int] = None) -> multiprocessing.shared_memory.SharedMemory:
        """Gets or creates the shared memory block."""
        size_needed = required_size or self.initial_size_bytes
        
        try:
            # Try to connect to existing shared memory
            shm = multiprocessing.shared_memory.SharedMemory(name=self.shared_memory_name)
            logger.info(f"🔗 Connected to existing shared memory: {self.shared_memory_name} (size: {shm.size} bytes)")
            # Check if we need more space
            if required_size and shm.size < required_size:
                logger.info(f"📈 Shared memory too small ({shm.size} < {required_size}), recreating...")
                shm.close()
                shm.unlink()
                raise FileNotFoundError("Recreating due to insufficient size")
            return shm
            
        except FileNotFoundError:
            # Create new shared memory
            logger.info(f"🆕 Creating new shared memory: {self.shared_memory_name} (size: {size_needed} bytes)")
            shm = multiprocessing.shared_memory.SharedMemory(
                name=self.shared_memory_name,
                create=True,
                size=size_needed
            )
            
            # Initialize header (size = 0)
            struct.pack_into('I', shm.buf, 0, 0)
            
            return shm
    
    def serialize_index_data(self, index: Any, id_mapping: Dict[int, str], 
                           piece_id_mapping: Dict[str, int]) -> bytes:
        """Serializes the FAISS index and mappings."""
        try:
            import faiss
            import io
            
            # Serialize FAISS index
            index_data = io.BytesIO()
            faiss.write_index(index, faiss.PyCallbackIOWriter(
                lambda data: index_data.write(data)
            ))
            index_bytes = index_data.getvalue()
            
            # Create complete data structure
            data_structure = {
                'index_bytes': index_bytes,
                'id_mapping': id_mapping,
                'piece_id_mapping': piece_id_mapping,
                'timestamp': time.time(),
                'vector_dimension': index.d if hasattr(index, 'd') else 768,
                'version': int(time.time() * 1000),  # Microsecond precision version
                'force_invalidate': True  # Force other processes to refresh
            }
            
            # Serialize everything with pickle
            serialized_data = pickle.dumps(data_structure)
            
            logger.info(f"📦 Serialized data: index={len(index_bytes)} bytes, "
                        f"mappings={len(id_mapping)} elements, total={len(serialized_data)} bytes")
            
            return serialized_data
            
        except Exception as e:
            logger.error(f"❌ Error serializing index data: {e}")
            raise
    
    def deserialize_index_data(self, data: bytes) -> Tuple[Any, Dict[int, str], Dict[str, int]]:
        """Deserializes the FAISS index and mappings."""
        try:
            import faiss
            import io
            
            # Deserialize data structure
            data_structure = pickle.loads(data)
            
            # Rebuild FAISS index
            index_bytes = data_structure['index_bytes']
            index_stream = io.BytesIO(index_bytes)
            index = faiss.read_index(faiss.PyCallbackIOReader(
                lambda n: index_stream.read(n)
            ))
            
            id_mapping = data_structure['id_mapping']
            piece_id_mapping = data_structure['piece_id_mapping']
            
            logger.info(f"📤 Deserialized data: vectors={index.ntotal}, "
                        f"mappings={len(id_mapping)} elements")
            
            return index, id_mapping, piece_id_mapping
            
        except Exception as e:
            logger.error(f"❌ Error deserializing index data: {e}")
            raise
    
    def write_to_shared_memory(self, index: Any, id_mapping: Dict[int, str], 
                              piece_id_mapping: Dict[str, int]) -> None:
        """Writes the index and mappings to shared memory."""
        with self._local_lock:
            try:
                # Serialize data
                serialized_data = self.serialize_index_data(index, id_mapping, piece_id_mapping)
                data_size = len(serialized_data)
                total_size_needed = self.header_size + data_size
                
                logger.info(f"💾 Writing to shared memory: {data_size} bytes of data")
                
                # Get or create shared memory with the required size
                if self.shared_memory is None or self.shared_memory.size < total_size_needed:
                    if self.shared_memory:
                        self.shared_memory.close()
                    self.shared_memory = self._get_or_create_shared_memory(total_size_needed * 2)  # Margen extra
                
                # Write with lock for synchronization
                with self.lock:
                    # Write data size
                    struct.pack_into('I', self.shared_memory.buf, 0, data_size)
                    
                    # Write serialized data
                    self.shared_memory.buf[self.header_size:self.header_size + data_size] = serialized_data
                    
                logger.info("Data successfully written to shared memory")
                
            except Exception as e:
                logger.error(f"❌ Error writing to shared memory: {e}")
                raise
    
    def read_from_shared_memory(self) -> Optional[Tuple[Any, Dict[int, str], Dict[str, int]]]:
        """Reads the index and mappings from shared memory."""
        with self._local_lock:
            try:
                # Connect to existing shared memory
                if self.shared_memory is None:
                    self.shared_memory = self._get_or_create_shared_memory()
                
                # Read with lock for synchronization
                with self.lock:
                    # Read data size
                    data_size = struct.unpack_from('I', self.shared_memory.buf, 0)[0]
                    
                    if data_size == 0:
                        logger.info("📭 No data in shared memory")
                        return None
                    logger.info(f"📖 Reading from shared memory: {data_size} bytes")
                    # Read serialized data
                    serialized_data = bytes(self.shared_memory.buf[self.header_size:self.header_size + data_size])
                
                # Deserialize data
                return self.deserialize_index_data(serialized_data)
                
            except FileNotFoundError:
                logger.info("📭 Shared memory does not exist yet")
                return None
            except Exception as e:
                logger.error(f"❌ Error reading from shared memory: {e}")
                return None
    
    def cleanup(self) -> None:
        """Cleans up shared memory resources."""
        try:
            if self.shared_memory:
                logger.info(f"🧹 Cleaning up shared memory: {self.shared_memory_name}")
                self.shared_memory.close()
                # Note: We do not call unlink() automatically to allow other processes to access
                
        except Exception as e:
            logger.error(f"❌ Error cleaning shared memory: {e}")
    
    def force_cleanup(self) -> None:
        """Forces full cleanup including unlink."""
        try:
            if self.shared_memory:
                self.shared_memory.close()
                self.shared_memory.unlink()
                logger.info(f"🗑️ Shared memory completely deleted: {self.shared_memory_name}")
                
        except Exception as e:
            logger.error(f"❌ Error in forced cleanup: {e}")

