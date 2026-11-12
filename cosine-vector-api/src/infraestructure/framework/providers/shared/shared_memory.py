import time
import logging
from typing import Optional, Tuple, Dict, Any
import multiprocessing.shared_memory
import struct
import pickle
import io
import faiss

from src.infraestructure.framework.config.config_root_module import get_app_config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SharedMemoryFaissManager:
    """
    Lector de índice FAISS desde memoria compartida, con reconexión automática.
    Comportamiento seguro: limita reintentos y aplica backoff para evitar bucles
    infinitos cuando la memoria compartida no existe.
    """
    def __init__(self, retry_interval: float = 1.0, max_retries: Optional[int] = 5, timeout: Optional[float] = None, backoff_factor: float = 1.5, reconnect_cooldown: float = 5.0):
        # Nombre de la memoria compartida (viene de configuración)
        self.shared_memory_name = get_app_config().faiss_shared_memory_name
        # Parámetros para reintentos
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
        # Cooldown (segundos) tras un fallo definitivo antes de reintentar
        self.reconnect_cooldown = reconnect_cooldown

        self.shared_memory: Optional[multiprocessing.shared_memory.SharedMemory] = None
        self.lock = multiprocessing.Lock()
        self.header_size = 4  # 4 bytes para tamaño de datos

        # Estado interno de reconexión
        self._connect_failed: bool = False
        self._last_failed_time: float = 0.0

        # Intentar conectar (pero no bloquear indefinidamente)
        self._connect()

    def _connect(self):
        """Intenta conectar a memoria compartida; respeta max_retries/timeout y aplica backoff.

        Si no logra conectarse, registra el fallo y vuelve rápidamente (no lanza excepción).
        Esto evita que llamadores queden bloqueados en un bucle infinito.
        """
        start = time.time()
        attempt = 0
        interval = self.retry_interval

        while True:
            try:
                self.shared_memory = multiprocessing.shared_memory.SharedMemory(name=self.shared_memory_name)
                logger.info(f"🔗 Conectado a memoria compartida: {self.shared_memory_name} (size: {self.shared_memory.size} bytes)")
                # Reset state on success
                self._connect_failed = False
                self._last_failed_time = 0.0
                return
            except FileNotFoundError:
                attempt += 1
                elapsed = time.time() - start

                # Determine if we should stop retrying
                give_up = False
                if self.max_retries is not None and attempt >= self.max_retries:
                    give_up = True
                if self.timeout is not None and elapsed >= self.timeout:
                    give_up = True

                if give_up:
                    logger.error(f"❌ No se pudo conectar a memoria compartida '{self.shared_memory_name}' después de {attempt} intentos ({elapsed:.1f}s). Se suspenderán reintentos temporales.")
                    self._connect_failed = True
                    self._last_failed_time = time.time()
                    self.shared_memory = None
                    return

                # Otherwise log and sleep with backoff
                logger.warning(f"📭 Memoria {self.shared_memory_name} no existe, reintentando (intento {attempt})...")
                time.sleep(interval)
                interval = min(interval * self.backoff_factor, 30.0)
                # Loop continues until success or give_up

    def _reconnect_if_needed(self):
        """Reconecta si la memoria fue destruida y recreada."""
        # If we recently failed to connect, avoid hammering repeated reconnects;
        # only allow a new reconnect attempt after reconnect_cooldown seconds.
        try:
            if self.shared_memory is not None:
                _ = self.shared_memory.buf[:1]  # acceso mínimo para verificar
                return
            else:
                raise ValueError("Shared memory is None")
        except (FileNotFoundError, ValueError, BufferError):
            # If we gave up recently, check cooldown
            now = time.time()
            if self._connect_failed and (now - self._last_failed_time) < self.reconnect_cooldown:
                # Skip reconnect attempt to avoid spamming logs
                return

            logger.warning("♻️ Memoria perdida o inaccesible, intentando reconectar...")
            if self.shared_memory:
                try:
                    self.shared_memory.close()
                except Exception:
                    pass
            # Attempt to reconnect (this will respect max_retries/timeout and set _connect_failed)
            self._connect()

    def read(self) -> Optional[Tuple[Any, Dict[int, str], Dict[str, int]]]:
        """
        Lee datos del índice FAISS desde la memoria.
        Devuelve: (index, id_mapping, piece_id_mapping)
        """
        self._reconnect_if_needed()

        if self.shared_memory is None:
            logger.error("❌ No se pudo conectar a la memoria compartida.")
            return None

        with self.lock:
            data_size = struct.unpack_from('I', self.shared_memory.buf, 0)[0]
            if data_size == 0:
                return None
            serialized_data = bytes(self.shared_memory.buf[self.header_size:self.header_size + data_size])

        return self._deserialize(serialized_data)

    def _flush(self):
        """Flush the shared memory."""
        self._reconnect_if_needed()

        if self.shared_memory is None:
            logger.error("❌ No se pudo conectar a la memoria compartida.")
            return

        with self.lock:
            self.shared_memory.buf[:1] = struct.pack('I', 0)  # Reset data size
            logger.info("🗑️ [FLUSH] Shared memory flushed.")

    def _deserialize(self, data: bytes) -> Tuple[Any, Dict[int, str], Dict[str, int]]:
        data_structure = pickle.loads(data)
        index_stream = io.BytesIO(data_structure['index_bytes'])
        index = faiss.read_index(faiss.PyCallbackIOReader(lambda n: index_stream.read(n)))
        return index, data_structure['id_mapping'], data_structure['piece_id_mapping']

    def close(self):
        if self.shared_memory:
            try:
                self.shared_memory.close()
            except Exception:
                pass
            self.shared_memory = None

    def write(self, index: Any, id_mapping: Dict[int, str], piece_id_mapping: Dict[str, int]) -> bool:
        """
        Write data to shared memory.
        Creates shared memory if it doesn't exist, or overwrites existing data.
        
        Args:
            index: FAISS index object
            id_mapping: Mapping from index IDs to piece IDs  
            piece_id_mapping: Mapping from piece IDs to index IDs
            
        Returns:
            True if write was successful, False otherwise
        """
        try:
            # Serialize the data
            serialized_data = self._serialize(index, id_mapping, piece_id_mapping)
            data_size = len(serialized_data)
            total_size = self.header_size + data_size
            
            logger.info(f"🔄 [WRITE] Serializing data: {data_size} bytes for shared memory")
            
            # Create or recreate shared memory with the right size
            if self.shared_memory:
                try:
                    self.shared_memory.close()
                except Exception:
                    pass
                self.shared_memory = None
            
            # Try to unlink existing shared memory first (in case it exists)
            try:
                existing_shm = multiprocessing.shared_memory.SharedMemory(name=self.shared_memory_name)
                existing_shm.close()
                existing_shm.unlink()
                logger.info(f"🗑️ [WRITE] Removed existing shared memory: {self.shared_memory_name}")
            except FileNotFoundError:
                # Good, it didn't exist
                pass
            except Exception as e:
                logger.warning(f"⚠️ [WRITE] Error removing existing shared memory: {e}")
            
            # Create new shared memory
            self.shared_memory = multiprocessing.shared_memory.SharedMemory(
                name=self.shared_memory_name,
                create=True,
                size=total_size
            )
            
            logger.info(f"✅ [WRITE] Created shared memory: {self.shared_memory_name} ({total_size} bytes)")
            
            # Write data
            with self.lock:
                # Write header (data size)
                struct.pack_into('I', self.shared_memory.buf, 0, data_size)
                # Write serialized data
                self.shared_memory.buf[self.header_size:self.header_size + data_size] = serialized_data
            
            vector_count = index.ntotal if index else 0
            logger.info(f"✅ [WRITE] Successfully wrote to shared memory: {vector_count} vectors, {len(id_mapping)} mappings")
            return True
            
        except Exception as e:
            logger.error(f"❌ [WRITE] Error writing to shared memory: {e}")
            import traceback
            logger.error(f"❌ [WRITE] Traceback: {traceback.format_exc()}")
            return False
    
    def _serialize(self, index: Any, id_mapping: Dict[int, str], piece_id_mapping: Dict[str, int]) -> bytes:
        """Serialize FAISS index and mappings to bytes."""
        try:
            # Serialize the index
            index_stream = io.BytesIO()
            faiss.write_index(index, faiss.PyCallbackIOWriter(lambda data: index_stream.write(data)))
            index_bytes = index_stream.getvalue()
            
            # Create data structure
            data_structure = {
                'index_bytes': index_bytes,
                'id_mapping': id_mapping,
                'piece_id_mapping': piece_id_mapping
            }
            
            # Serialize with pickle
            return pickle.dumps(data_structure)
            
        except Exception as e:
            logger.error(f"❌ [SERIALIZE] Error serializing data: {e}")
            raise
