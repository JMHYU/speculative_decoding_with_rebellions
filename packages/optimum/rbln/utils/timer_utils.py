from datetime import datetime

from .logging import get_logger


logger = get_logger()


def rbln_timer(print_name):
    def decorator(function):
        def wrapper(*args, **kwargs):
            tick = datetime.now()
            result = function(*args, **kwargs)
            logger.debug(f"{print_name}. Elasped time: {str(datetime.now() - tick)[:7]}")
            return result

        return wrapper

    return decorator
