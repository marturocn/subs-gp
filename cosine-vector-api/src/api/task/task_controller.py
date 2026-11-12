from fastapi import APIRouter, Path, status
from dishka.integrations.fastapi import inject
from inspect import getdoc


from api.task.dto.task_id_dto import TaskIdDto
from common.core.observable_resolver import resolve_observable_to_single_result
from infraestructure.framework.validators.validate_body import ValidateBody
from infraestructure.framework.validators.validate_params import ValidateParams
from src.domain.task.use_cases import UpdateTaskUseCase, UpdateTaskCommand, CreateTaskUseCase, CreateTaskCommand, GetTasksUseCase
from src.domain.task.models.entities.task import Task

from .dto.create_task_dto import CreateTaskDto
from .dto.update_task_dto import UpdateTaskDto

# Constants
INTERNAL_SERVER_ERROR_DESC = "Internal Server Error"
# when TaskController is instantiated.

@inject
class TaskController:
    # Remove the `router` attribute from here.
    # It will be assigned to the instance of the controller.

    def __init__(
        self,
        get_tasks_use_case: GetTasksUseCase,
        create_task_use_case: CreateTaskUseCase,
        update_task_use_case: UpdateTaskUseCase,
    ):
        self._get_tasks_use_case = get_tasks_use_case
        self._create_task_use_case = create_task_use_case
        self._update_task_use_case = update_task_use_case
        self.router = APIRouter(prefix="/tasks", tags=["tasks"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/",
            self.get_tasks,
            methods=["GET"],
            description=getdoc(GetTasksUseCase),
            responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": INTERNAL_SERVER_ERROR_DESC}},
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/",
            self.create_task,
            methods=["POST"],
            description=getdoc(CreateTaskUseCase),
            responses={
                status.HTTP_400_BAD_REQUEST: {"description": "Bad Request"},
                status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation Error"},
                status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": INTERNAL_SERVER_ERROR_DESC},
            },
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/{id}",
            self.update_task,
            methods=["PATCH"],
            description=getdoc(UpdateTaskUseCase),
            responses={
                status.HTTP_400_BAD_REQUEST: {"description": "Bad Request"},
                status.HTTP_404_NOT_FOUND: {"description": "Task Not Found"},
                status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation Error"},
                status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": INTERNAL_SERVER_ERROR_DESC},
            },
            status_code=status.HTTP_200_OK,
        )

    # Note: No @inject decorator on individual methods anymore
    async def get_tasks(self):
        """
        Obtiene todas las tareas.
        """
        response_observable = self._get_tasks_use_case.execute(None)
        get_tasks_response = await resolve_observable_to_single_result(response_observable)
        return get_tasks_response

    async def create_task(
        self,
        command: CreateTaskDto = ValidateBody(CreateTaskDto)(),
    ) -> Task:
        """
        Crea una nueva tarea.
        """
        cmd = CreateTaskCommand(
            title=command.title,
            description=command.description,
            is_completed=command.is_completed
        )
        observable = self._create_task_use_case.execute(cmd)
        return await resolve_observable_to_single_result(observable)

    async def update_task(
        self,
        id: int = Path(..., description="ID de la tarea"),
        dto: UpdateTaskDto = ValidateBody(UpdateTaskDto)(),
    ) -> Task:
        """
        Actualiza una tarea existente.
        """
        command = UpdateTaskCommand(
            id=str(id),
            title=dto.title,
            description=dto.description,
            is_completed=dto.is_completed
        )
        observable = self._update_task_use_case.execute(command)
        return await resolve_observable_to_single_result(observable)