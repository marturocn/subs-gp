# Define a TypeVar for the generic observable
import asyncio
from typing import TypeVar

from reactivex import Observable
from reactivex.operators import first # 'first' operator to get the first emitted value


_T = TypeVar("_T")

# Refactored function to resolve an Observable to a single result
async def resolve_observable_to_single_result(observable: Observable[_T]) -> _T:
    """
    Consumes a reactivex Observable, waits for its first emitted value,
    and returns it as an awaitable result. Handles errors.
    """
    loop = asyncio.get_event_loop()
    future: asyncio.Future[_T] = loop.create_future()

    # Subscribe to the observable, taking only the first emitted value
    # and setting the future's result or exception
    observable.pipe(
        first() # Ensure we only take the first element, as typically expected from a use case result
    ).subscribe(
        on_next=lambda value: future.set_result(value),
        on_error=lambda error: future.set_exception(error)
    )

    return await future
