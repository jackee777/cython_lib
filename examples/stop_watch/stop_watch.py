from functools import wraps
import time

def calc_time(function) :
    @wraps(function)
    def wrapper(*args, **kargs) :
        start = time.perf_counter()
        result = function(*args,**kargs)
        process_time =  time.perf_counter() - start
        print("{} spends {} ms the "
              "calculaton".format(function.__name__, process_time))
        return result
    return wrapper