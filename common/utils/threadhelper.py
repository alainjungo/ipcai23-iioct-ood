import multiprocessing as mp
import traceback

NUM_WORKERS = 2

pool = None


# To use a pool might not be the ideal solution
# I used threads before but they did not parallelize properly (almost no time gain)
def do_work(fn, *args, in_background=True):
    if in_background:
        global pool
        if pool is None:
            pool = mp.Pool(NUM_WORKERS)

        def clb(e: BaseException):
            traceback.print_exception(type(e), e, e.__traceback__)
        pool.apply_async(func=fn, args=args, error_callback=clb)
    else:
        fn(*args)


def join_all():
    if pool is not None:
        pool.close()
        pool.join()
