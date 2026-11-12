"""
Logger global configurado usando la configuración de AppConfigService.
"""
import logging
import os
from typing import Optional

_logger: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    """
    Get or create the global logger instance.
    Uses lazy initialization to ensure .env is loaded first.
    """
    global _logger
    
    if _logger is None:
        # Importar aquí para evitar problemas de inicialización
        from src.infraestructure.framework.config.files.app_config import app_config
        from src.infraestructure.framework.config.services.app_config_service import AppConfigService
        
        # Cargar configuración
        config_dict = app_config()
        config_service = AppConfigService(config_dict)
        
        # Obtener nivel de log con fallback
        log_level = config_service.get_optional("log_level", "INFO")
        if log_level is None or str(log_level).upper() == "NONE":
            log_level = "INFO"
        log_level = str(log_level).upper()
        
        # Crear logger global
        _logger = logging.getLogger("LOGGER")
        _logger.setLevel(log_level)
        
        # Limpiar handlers existentes
        _logger.handlers.clear()
        
        # Formato similar a pino-pretty
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.propagate = False
        
        # Opcional: cambiar formato si no es producción
        if not config_service.is_production:
            # Puedes personalizar aún más el formato aquí si lo deseas
            pass
    
    return _logger

# Mantener compatibilidad con el código existente
logger = get_logger()

# Uso:
# from src.infraestructure.framework.providers.logger.logger import logger
# logger.info("Mensaje informativo")
# logger.error("Mensaje de error")
