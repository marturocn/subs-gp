# Assuming your project uses 'reactivex'
from typing import List
import reactivex # You might need this if you use reactivex.from_ or other reactivex top-level functions
from reactivex import Observable # <--- THIS IS THE CORRECT IMPORT FOR REACTIVEX
from reactivex import operators as op

from src.domain.task.use_cases.get_tasks_case import GetTasksUseCase, GetTasksResponse, TaskResponse
from src.domain.task.repositories.task_repository import TaskRepository
from src.domain.task.models.entities.task import Task
from src.infraestructure import logger

class GetTasksUseCaseImpl(GetTasksUseCase):
    """
    Implementation of GetTasksUseCase.
    """

    def __init__(self, repository: TaskRepository):
        """
        Initialize the use case with a task repository.

        Args:
            repository: Task repository for data persistence
        """
        self.repository = repository
        logger.debug("GetTasksUseCaseImpl initialized with repository: %s", type(repository).__name__)

    def execute(self, command: None) -> Observable[GetTasksResponse]:
        logger.info("Executing GetTasksUseCaseImpl.execute()")
        logger.debug("Calling repository.find() to retrieve tasks")

        # Ensure that self.repository.find() is correctly typed to return Observable[List[Task]]
        tasks_observable: Observable[List[Task]] = self.repository.find()
        logger.debug("Retrieved tasks observable: %s", tasks_observable)

        def map_tasks_to_response(tasks: List[Task]) -> GetTasksResponse:
            """
            Maps a list of Task entities to a GetTasksResponse DTO.
            This function will be applied by the RxPy map operator.
            """
            task_responses = [
                TaskResponse(
                    id=task.id or "",
                    title=task.title,
                    description=task.description,
                    is_completed=task.is_completed
                )
                for task in tasks
            ]
            return GetTasksResponse(tasks=task_responses, total=len(task_responses))

        return tasks_observable.pipe(
            op.map(map_tasks_to_response)
        )