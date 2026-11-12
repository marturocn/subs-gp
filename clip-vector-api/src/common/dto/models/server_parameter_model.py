"""
Modelo para parámetros de servidor.
"""
from typing import Any, Dict, Optional


class ServerParameterModel:
    """
    Modelo para parámetros de servidor equivalente al modelo TypeScript.
    """
    
    def __init__(self, key: str = "", message: str = "", payload: Optional[Dict[str, Any]] = None):
        """
        Inicializa el modelo de parámetro del servidor.
        
        Args:
            key: Clave del parámetro
            message: Mensaje del parámetro
            payload: Diccionario opcional con datos adicionales para asignar
        """
        self.key = key
        self.message = message
        
        # Si se proporciona un payload, asignar sus valores
        if payload:
            for attr_key, value in payload.items():
                if hasattr(self, attr_key):
                    setattr(self, attr_key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el modelo a diccionario.
        
        Returns:
            Diccionario con los datos del modelo
        """
        return {
            "key": self.key,
            "message": self.message
        }
    
    def __repr__(self) -> str:
        """Representación string del objeto."""
        return f"ServerParameterModel(key='{self.key}', message='{self.message}')"
    
    def __eq__(self, other: object) -> bool:
        """Comparación de igualdad."""
        if not isinstance(other, ServerParameterModel):
            return False
        return self.key == other.key and self.message == other.message
