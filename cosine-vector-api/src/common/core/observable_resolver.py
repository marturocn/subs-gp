# Define a TypeVar for the generic observable
import asyncio
import logging
from typing import TypeVar

from reactivex import Observable
from reactivex.operators import first


_T = TypeVar("_T")
logger = logging.getLogger(__name__)

# Refactored function to resolve an Observable to a single result
async def resolve_observable_to_single_result(observable: Observable[_T]) -> _T:
    """
    Consumes a reactivex Observable, waits for its first emitted value,
    and returns it as an awaitable result. Handles errors and timeouts.
    
    Args:
        observable: The Observable to resolve
    
    Note: Uses asyncio.timeout context manager internally with default 60s timeout
    """
    loop = asyncio.get_event_loop()
    future: asyncio.Future[_T] = loop.create_future()

    def on_next(value):
        if not future.done():
            logger.info(f"✅ Observable resolved successfully with value type: {type(value)}")
            future.set_result(value)

    def on_error(error):
        if not future.done():
            logger.error(f"❌ Observable failed with error: {type(error).__name__}: {error}")
            future.set_exception(error)

    def on_completed():
        if not future.done():
            logger.warning("⚠️ Observable completed without emitting any value")
            future.set_exception(RuntimeError("Observable completed without emitting any value"))

    logger.info("🔄 Observable resolver: Starting subscription with 60s timeout")
    
    # Subscribe to the observable, taking only the first emitted value
    # and setting the future's result or exception
    observable.pipe(
        first() # Ensure we only take the first element, as typically expected from a use case result
    ).subscribe(
        on_next=on_next,
        on_error=on_error,
        on_completed=on_completed
    )

    logger.info("🔄 Observable resolver: Subscription created, waiting for result...")
    
    try:
        # Wait for the result with timeout using context manager
        async with asyncio.timeout(30.0):
            result = await future
        logger.info(f"🎉 Observable resolver: Successfully resolved with result type: {type(result)}")
        return result
    except asyncio.TimeoutError:
        logger.error("❌ Observable resolution timed out after 60 seconds")
        future.cancel()
        raise RuntimeError("Operation timed out after 60 seconds")
