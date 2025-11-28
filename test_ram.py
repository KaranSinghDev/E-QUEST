"""
test_memory_profiler.py

A thorough test suite to validate memory_profiler on WSL2.

Usage:
    # install dependencies first (run once)
    pip install memory_profiler psutil numpy

    # Basic line-by-line memory profiling (human readable)
    python -m memory_profiler test_memory_profiler.py

    # Save output to file
    python -m memory_profiler test_memory_profiler.py > memprof_output.txt

    # Alternatively, use `mprof` (part of memory_profiler) to record and plot:
    # pip install matplotlib
    # mprof run --python python test_memory_profiler.py
    # mprof plot
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp

from memory_profiler import profile
import psutil

# Optional: Numpy used for large contiguous allocations
try:
    import numpy as np
except Exception:
    np = None


LOG_CSV = "memprof_test_log.csv"


def log_process_memory(tag: str):
    """Record an instant snapshot of the current process memory (RSS) using psutil."""
    p = psutil.Process(os.getpid())
    mem_mb = p.memory_info().rss / 1024 / 1024
    line = f"{time.time()},{tag},{mem_mb:.3f}\n"
    with open(LOG_CSV, "a") as f:
        f.write(line)
    print(f"[PSUTIL] {tag}: RSS = {mem_mb:.3f} MB")


@profile
def grow_python_list(n_items=5_000_00, chunk=50_000):
    """
    Repeatedly append to a Python list to see memory changes per-line.
    - n_items: approximate final size
    - chunk: how many appends per iteration (so memory_profiler shows per-loop changes)
    """
    log_process_memory("before_grow_list")
    data = []
    for i in range(0, n_items, chunk):
        # allocate a chunk of small objects (strings) to grow memory gradually
        chunk_list = [str(j) * 20 for j in range(chunk)]
        data.extend(chunk_list)
        # small sleep to let profiler sample in some runs
        time.sleep(0.01)
    log_process_memory("after_grow_list")
    # keep reference so memory remains allocated while we measure
    return data


@profile
def allocate_numpy_array(shape=(50_000, 100)):
    """
    Allocate a big numpy array (if numpy is available).
    This shows a large contiguous allocation (useful for ML/data pipelines).
    """
    log_process_memory("before_numpy")
    if np is None:
        print("NumPy not available — skipping numpy test.")
        return None
    arr = np.ones(shape, dtype=np.float64)  # high memory footprint
    log_process_memory("after_numpy_alloc")
    # touch the array a bit
    arr[0, 0] = 123.456
    time.sleep(0.5)
    log_process_memory("after_numpy_touch")
    return arr


@profile
def simulate_memory_leak(n=200_000):
    """
    Simulate a common 'leak' pattern: keep growing a dict with references.
    This is to show memory_profiler capturing allocations that are not freed.
    """
    log_process_memory("before_leak")
    leak = {}
    for i in range(n):
        # store small lists/tuples in a dict keyed by i
        leak[i] = (i, "x" * 50)
        if i % 50_000 == 0:
            time.sleep(0.01)
    log_process_memory("after_leak")
    # return to keep it alive
    return leak


def child_worker_alloc(size_mb=200):
    """Child process target: allocate a bytes object of given size in MB and sleep."""
    b = bytearray(size_mb * 1024 * 1024)
    # print child's own memory snapshot so you can cross-check
    p = psutil.Process(os.getpid())
    print(f"[CHILD] pid={os.getpid()} RSS={p.memory_info().rss / 1024 / 1024:.2f} MB")
    time.sleep(3)  # stay alive briefly so parent can inspect


@profile
def spawn_multiprocessing_child(size_mb=150):
    """
    Spawn a child process that allocates memory. This demonstrates a child process allocation.
    Note: memory_profiler profiling is per-process; the parent's run will not show child's internal line-by-line memory.
    """
    log_process_memory("before_spawn_child")
    p = mp.Process(target=child_worker_alloc, args=(size_mb,))
    p.start()
    time.sleep(1)  # give the child time to allocate
    # check child's memory from parent using psutil
    try:
        child_ps = psutil.Process(p.pid)
        print(f"[PARENT] Observed child pid={p.pid} RSS={child_ps.memory_info().rss / 1024 / 1024:.2f} MB")
    except Exception as ex:
        print("Could not inspect child process memory:", ex)
    p.join()
    log_process_memory("after_spawn_child")


@profile
def spawn_subprocess_alloc(size_mb=100):
    """
    Launch an external Python subprocess that allocates memory. This is useful to test profiler vs. external processes.
    We use a short inline Python snippet to allocate memory and print its RSS.
    """
    log_process_memory("before_subprocess")
    # Inline script: allocate `size_mb` MB, print RSS via psutil in child
    snippet = (
        "import time, psutil, os; "
        f"b = bytearray({size_mb} * 1024 * 1024); "
        "p=psutil.Process(os.getpid()); "
        "print(f\"[SUBPROC] pid={os.getpid()} RSS={p.memory_info().rss / 1024 / 1024:.2f} MB\"); "
        "time.sleep(2)"
    )
    subprocess.run([sys.executable, "-c", snippet], check=True)
    log_process_memory("after_subprocess")


def prepare_log():
    """Initialize CSV log."""
    with open(LOG_CSV, "w") as f:
        f.write("ts,tag,rss_mb\n")


def main():
    prepare_log()
    print("== Memory profiler test suite starting ==")
    log_process_memory("start_main")
    # 1) grow a Python list
    data = grow_python_list(n_items=150_000, chunk=15_000)

    # 2) allocate a numpy array (if available)
    arr = allocate_numpy_array(shape=(30_000, 50))  # moderate size

    # 3) simulate leak
    leak = simulate_memory_leak(n=80_000)

    # 4) spawn multiprocessing child that allocates
    spawn_multiprocessing_child(size_mb=120)

    # 5) spawn a subprocess and observe memory
    spawn_subprocess_alloc(size_mb=80)

    # Free some references and force a short pause so profiler may observe release
    del data
    del arr
    del leak
    time.sleep(1)
    log_process_memory("end_main")
    print("== Test suite complete ==")


if __name__ == "__main__":
    main()
