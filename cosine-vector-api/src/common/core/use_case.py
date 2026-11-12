from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from reactivex import Observable # <--- THIS IS THE CORRECT IMPORT FOR REACTIVEX


I = TypeVar('I')
O = TypeVar('O')

class UseCase(ABC, Generic[I, O]):
    """
    Abstract base class for use cases.
    Represents a generic use case with input and output types.
    """
    @abstractmethod
    def execute(self, command: I) -> Observable[O]:
        """
        Executes the use case logic with the given command.
        Args:
            command (I): The input command or data for the use case.
        Returns:
            Observable[O]: An observable emitting the result of the use case.
        """
        pass