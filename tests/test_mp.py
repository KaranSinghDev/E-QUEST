# File: run_profiler_test.py

import subprocess
import sys

def main():
    """
    This script perfectly mimics how run_benchmark.py calls its children.
    It runs the memory test script as a subprocess and captures BOTH
    stdout and stderr to reveal exactly what memory-profiler is outputting
    and where it's sending it.
    """
    
    # The script we want to test
   
    target_script = "tests/memory_profile.py"
    
    # The exact command that run_benchmark.py uses
    command = [sys.executable, "-m", "memory_profiler", target_script]
    
    print("="*60)
    print(f"🚀 Running command: {' '.join(command)}")
    print("="*60)
    
    try:
        # Execute the command, capturing all output
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        print("\n--- CAPTURED STDOUT ---")
        if result.stdout:
            print(result.stdout)
        else:
            print("<stdout was empty>")
            
        print("\n--- CAPTURED STDERR ---")
        if result.stderr:
            print(result.stderr)
        else:
            print("<stderr was empty>")
            
        print("\n" + "="*60)
        print("✅ TEST COMPLETE")
        print("="*60)

    except FileNotFoundError:
        print(f"ERROR: Could not find the test script '{target_script}'. Make sure it's in the same directory.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: The subprocess failed.")
        print("\n--- FAILED STDOUT ---")
        print(e.stdout)
        print("\n--- FAILED STDERR ---")
        print(e.stderr)

if __name__ == "__main__":
    main()