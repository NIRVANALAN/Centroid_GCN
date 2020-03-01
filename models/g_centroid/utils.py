import time


def timeit(fn):
    def get_time(*args, **kwargs):
        start = time.time()
        output = fn(*args, **kwargs)
        print(f'Time taken in {fn.__name__}: {time.time() - start:.7}')
        return output
    return get_time
