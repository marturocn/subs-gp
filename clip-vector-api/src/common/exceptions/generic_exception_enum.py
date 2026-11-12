from enum import Enum

# Enum de errores genéricos
class GenericExceptionsEnum(str, Enum):
    IS_NUMBER = "$property,$property must be number"
    IS_STRING = "$property,$property must be string"
    IS_NOT_EMPTY = "$property,$property must not be empty"
    IS_DEFINED = "$property,$property must be defined"
    IS_ARRAY = "$property,$property must be an array"
    IS_BOOLEAN = "$property,$property must be a boolean"


# Función para formatear mensajes de error
def format_exception_message(message_enum: GenericExceptionsEnum, field_name: str) -> str:
    return message_enum.value.replace("$property", field_name)
