import logging
import time

logger = logging.getLogger(__name__)


def time2str(time_used):
    gaps = [
        ("days", 86400000),
        ("h", 3600000),
        ("min", 60000),
        ("s", 1000),
        ("ms", 1),
    ]
    time_used *= 1000
    time_str = []
    for unit, gap in gaps:
        val = time_used // gap
        if val > 0:
            time_str.append("{}{}".format(int(val), unit))
            time_used -= val * gap
    if len(time_str) == 0:
        time_str.append("0ms")
    return " ".join(time_str)


def get_date():
    return time.strftime("%Y-%m-%d", time.localtime(time.time()))


def get_time(t=None):
    if t is None:
        t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def with_time(func, pretty_time=False):
    """
    Usage
    -----

    1. as a function decorator

    ``` python
    @with_time
    def func(...):
        ...
    result, cost_in_seconds = func(...)
    ```

    2. directly apply

    ``` python
    result, cost_in_seconds = with_time(func)(...)
    ```
    """

    def wrap_time(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        time_cost = time.time() - start
        if pretty_time:
            time_cost = time2str(time_cost)
        return res, time_cost

    return wrap_time


def timeit_logger(func):
    func_name = func.__qualname__

    def wrapped(*args, **kwargs):
        logger.opt(colors=True).info(
            "* Entering <u><blue>{func_name}</blue></u>...", func_name=func_name
        )
        func_wrapped = with_time(func, pretty_time=True)
        result, time_cost = func_wrapped(*args, **kwargs)
        logger.opt(colors=True).info(
            "* <u><blue>{func_name}</blue></u> finished. Time cost: {time_cost}",
            func_name=func_name,
            time_cost=time_cost,
        )
        return result

    return wrapped
