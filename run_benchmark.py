# (Just below the existing imports)

# --- Add the project root to the Python path ---
# This ensures that all subprocesses and modules can find 'src' and 'trackml'
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
# We also add the 'src' directory from the cloned trackml library
trackml_src_path = os.path.join(project_root, 'src', 'trackml-library', 'trackml')


# --- Let's try an even more robust method ---
# The core issue is that the subprocess does not inherit the Python path.
# The solution is to pass the path to the subprocess.
# Let's abandon the sys.path modification in the conductor.
# The problem is purely with the individual scripts.
import subprocess
import shutil
import pandas as pd
from datetime import datetime
import time
# --- Configuration for the Conductor Script ---
# This defines the two possible modes and the workflow sequence.
CONFIG_FULL_PATH = os.path.join('src', 'config_full.py')
CONFIG_SMOKE_TEST_PATH = os.path.join('src', 'config_smoke_test.py')
TARGET_CONFIG_PATH = os.path.join('src', 'config.py')

WORKFLOW_SCRIPTS = [
    "scripts/create_ml_dataset.py",
    "scripts/preprocess_ml_dataset.py",
    "scripts/analyze_scaling_classical.py",
    "scripts/analyze_scaling_quantum.py",
    "scripts/generate_report.py"
]

def generate_markdown_report():
    """Generates a final summary report in Markdown format."""
    print("\n" + "="*60)
    print("📝 GENERATING FINAL MARKDOWN REPORT 📝")
    print("="*60)
    
    try:
        classical_df = pd.read_csv(os.path.join('results', 'classical_results.csv'))
        quantum_df = pd.read_csv(os.path.join('results', 'quantum_results.csv'))
    except FileNotFoundError:
        print("❌ ERROR: Could not find results CSV files. Skipping report generation.")
        return

    # Extract key statistics from the largest run
    c_final = classical_df.iloc[-1]
    q_final = quantum_df.iloc[-1]

    report_content = f"""
# E-QUEST Framework: Final Analysis Report

*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

This report summarizes the benchmark results for the Classical MLP and Quantum VQC algorithms as executed by the E-QUEST framework.

---

## Analysis 1: Energy Scaling

This analysis compares the empirical energy consumption on current hardware (GPU) against the theoretical energy projections based on hardware-independent models (MAC operations for classical, gate counts for quantum). This provides a forecast for future sustainability.

![Energy Analysis](./1_energy_analysis.png)

---

## Analysis 2: Memory & Resource Scaling

This plot shows the empirical peak GPU memory usage during training. The theoretical space complexity for both algorithms with respect to the number of samples (`n`) is O(1), as the memory is dominated by the fixed-size model and data batches, not the total dataset size.

![Memory Analysis](./2_memory_analysis.png)

---

## Analysis 3: Model Performance

This plot compares the predictive accuracy (ROC AUC Score) of both models as the number of training samples increases. A score of 0.5 represents random guessing.

![Performance Analysis](./3_performance_analysis.png)

---

## Analysis 4: Time Scaling

This plot compares the empirical execution time against the theoretical time complexity, which is O(n) for both algorithms in this training paradigm.

![Time Analysis](./4_time_analysis.png)

---

## Summary of Key Metrics (at largest input size)

| Metric                   | Classical MLP              | Quantum VQC                |
|--------------------------|----------------------------|----------------------------|
| **Input Size**           | {c_final['input_size']:,}  | {q_final['input_size']:,}  |
| **Execution Time (s)**   | {c_final['gpu_train_time_s']:.2f} | {q_final['sim_time_gpu_s']:.2f} |
| **Peak Memory (MB)**     | {c_final['peak_memory_mb']:.2f}   | {q_final['peak_memory_mb']:.2f}   |
| **Accuracy (AUC)**       | {c_final['accuracy_auc']:.3f}     | {q_final['accuracy_auc']:.3f}     |
| **Circuit Depth**        | N/A                        | {q_final['circuit_depth']} |

"""
    report_path = os.path.join('results', 'final_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"✅ Final report saved successfully to '{report_path}'")

# --- REPLACE THE ENTIRE main() FUNCTION WITH THIS ---
# --- REPLACE YOUR ENTIRE main() FUNCTION WITH THIS ---
def main():
    """Main conductor function to run the entire workflow."""
    # --- Determine which mode to run ---
    is_smoke_test = '--smoke-test' in sys.argv
    source_config = CONFIG_SMOKE_TEST_PATH if is_smoke_test else CONFIG_FULL_PATH
    
    mode_message = "SMOKE TEST" if is_smoke_test else "FULL ANALYSIS"
    
    print("="*60)
    print(f"🚀 STARTING E-QUEST WORKFLOW IN **{mode_message}** MODE 🚀")
    print("="*60)

    try:
        print(f"-> Preparing configuration from '{source_config}'...")
        shutil.copyfile(source_config, TARGET_CONFIG_PATH)
        print("✅ Configuration is set.")

        # --- Execute the workflow step-by-step and collect stats ---
        dataset_path_to_use = None
        workflow_stats = [] # List to store dictionaries of stats

        for i, script_path in enumerate(WORKFLOW_SCRIPTS):
            print("\n" + "-"*60)
            print(f"--- [Step {i+1} of {len(WORKFLOW_SCRIPTS)}] Executing: {script_path} ---")
            print("-"*60)
            
            stage_start_time = time.perf_counter()
            
            command = [sys.executable, script_path]
            
            # --- START OF DEFINITIVE FIX ---
            # If this is NOT the first script, it needs the dataset path from the previous step.
            if i > 0: 
                if not dataset_path_to_use:
                    raise RuntimeError("Dataset path was not set by the initial script.")
                command.append(dataset_path_to_use)
            # --- END OF DEFINITIVE FIX ---

            subprocess.run(command, check=True)

            stage_end_time = time.perf_counter()
            elapsed_time = stage_end_time - stage_start_time
            
            workflow_stats.append({
                "script": script_path,
                "time_s": elapsed_time,
            })

            # --- START OF DEFINITIVE FIX ---
            # After any script runs, check if it produced a new path file.
            # This allows the preprocess script to hand off the new balanced path.
            if os.path.exists("current_dataset_path.txt"):
                with open("current_dataset_path.txt", "r") as f:
                    new_path = f.read().strip()
                if new_path != dataset_path_to_use:
                    dataset_path_to_use = new_path
                    print(f"  -> Conductor updated dataset path to: {dataset_path_to_use}")
            # --- END OF DEFINITIVE FIX ---

            print(f"✅ Step {i+1} completed successfully in {elapsed_time:.2f} seconds.")

        # --- The rest of the function (reporting and cleanup) remains the same ---
        
        # ... (Your existing code for generating the final report) ...
        # (Make sure to copy the full block from the last working version)
        
        # For completeness, here is the full reporting and cleanup block:
        print("\n" + "="*60)
        print("🎉 ENTIRE WORKFLOW COMPLETED SUCCESSFULLY! 🎉")
        
        data_file_size_mb = os.path.getsize(dataset_path_to_use) / (1024 * 1024) if dataset_path_to_use and os.path.exists(dataset_path_to_use) else 0
        results_dir_size_mb = sum(os.path.getsize(os.path.join('results', f)) for f in os.listdir('results')) / (1024 * 1024) if os.path.exists('results') else 0

        print("\n--- 📊 Workflow Resource Usage Report ---")
        print("-" * 50)
        print(f"{'Stage':<40} {'Time (s)':>10}")
        print("-" * 50)
        total_time = 0
        for stats in workflow_stats:
            print(f"{stats['script']:<40} {stats['time_s']:>10.2f}")
            total_time += stats['time_s']
        print("-" * 50)
        print(f"{'Total Workflow Time':<40} {total_time:>10.2f}")
        print("\n--- 💾 Disk Usage ---")
        print(f"  -> Final Dataset Size: {data_file_size_mb:.2f} MB")
        print(f"  - Results Directory Size: {results_dir_size_mb:.2f} MB")
        print("="*60)
        print(f"\n-> All plots and the final report are available in the 'results/' directory.")

    except FileNotFoundError:
        print("\n❌ CRITICAL ERROR: A required configuration file is missing.")
        print(f"   Please ensure both '{CONFIG_FULL_PATH}' and '{CONFIG_SMOKE_TEST_PATH}' exist.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("\n❌ CRITICAL ERROR: A script in the workflow failed to execute.")
        print("   Please check the error messages above to diagnose the issue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # --- Cleanup ---
        if os.path.exists(TARGET_CONFIG_PATH):
            os.remove(TARGET_CONFIG_PATH)
        if os.path.exists("current_dataset_path.txt"):
            os.remove("current_dataset_path.txt")
        print("\n-> Cleanup complete: Temporary configuration files removed.")

if __name__ == "__main__":
    main()
