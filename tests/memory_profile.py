# File: test_memory_profiler.py

import numpy as np
import time

def main():
    """
    A simple script that allocates a large block of memory to test the profiler.
    """
    print("--- Test Script Starting ---")
    
    # Allocate a 10000x10000 array of 8-byte floats.
    # This should be ~763 MiB of memory.
    try:
        print("Allocating a large NumPy array...")
        large_array = np.zeros((10000, 10000), dtype=np.float64)
        print(f"Successfully allocated array with shape: {large_array.shape}")
        
        # Hold the memory for a couple of seconds
        time.sleep(2)
        
    except MemoryError:
        print("ERROR: Not enough memory to allocate the large test array.")
        
    print("--- Test Script Finished ---")

if __name__ == "__main__":
    main()