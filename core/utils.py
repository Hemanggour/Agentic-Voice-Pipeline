import functools
import time


def time_it(func):
    """
    Decorator that prints the execution time of the function it decorates.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(
            f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute."
        )
        return result

    return wrapper
