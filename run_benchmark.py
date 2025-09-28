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

# --- Configuration for the Conductor Script ---
# This defines the two possible modes and the workflow sequence.
CONFIG_FULL_PATH = os.path.join('src', 'config_full.py')
CONFIG_SMOKE_TEST_PATH = os.path.join('src', 'config_smoke_test.py')
TARGET_CONFIG_PATH = os.path.join('src', 'config.py')

WORKFLOW_SCRIPTS = [
    "scripts/create_ml_dataset.py",
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


def main():
    """Main conductor function to run the entire workflow."""
        # --- ADDED: Ensure trackml is properly installed ---
    # This is a robust check for non-standard packages.
    try:
        import trackml
    except ImportError:
        print("="*60)
        print("⚠️  WARNING: 'trackml' library not found or not discoverable.")
        print("-> Attempting to install it in editable mode from GitHub...")
        # We need to find the path to the library source if it was cloned.
        # This is a very complex problem. Let's simplify.
        # A much simpler and mor
    
    # --- Determine which mode to run ---
    is_smoke_test = '--smoke-test' in sys.argv
    source_config = CONFIG_SMOKE_TEST_PATH if is_smoke_test else CONFIG_FULL_PATH
    
    mode_message = "SMOKE TEST" if is_smoke_test else "FULL ANALYSIS"
    
    print("="*60)
    print(f"🚀 STARTING E-QUEST WORKFLOW IN **{mode_message}** MODE 🚀")
    print("="*60)

    # --- This is the core logic to dynamically set the configuration ---
    # It copies the selected config (full or smoke) to the target `src/config.py`
    # so that all other scripts can import it without being changed.
    try:
        print(f"-> Preparing configuration from '{source_config}'...")
        shutil.copyfile(source_config, TARGET_CONFIG_PATH)
        print("✅ Configuration is set.")

        # --- Execute the workflow step-by-step ---
        for i, script_path in enumerate(WORKFLOW_SCRIPTS):
            print("\n" + "-"*60)
            print(f"--- [Step {i+1} of {len(WORKFLOW_SCRIPTS)}] Executing: {script_path} ---")
            print("-"*60)
            
            # We use subprocess.run with check=True.
            # This is robust and will automatically stop if a script fails.  # --- THIS IS THE DEFINITIVE FIX ---
# We use sys.executable to get the absolute path to the Python interpreter
# that is currently running THIS script (e.g., .../venv/Scripts/python.exe).
# This guarantees that the subprocess uses the exact same environment.
            subprocess.run([sys.executable, script_path], check=True)

            
            print(f"✅ Step {i+1} completed successfully.")

        # --- Generate the final report ---
        generate_markdown_report()

        print("\n" + "="*60)
        print("🎉 ENTIRE WORKFLOW COMPLETED SUCCESSFULLY! 🎉")
        print(f"-> All plots and the final report are available in the 'results/' directory.")
        print("="*60)

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
        # This block ensures that the temporary `src/config.py` is removed
        # even if the script fails, leaving the project in a clean state.
        if os.path.exists(TARGET_CONFIG_PATH):
            os.remove(TARGET_CONFIG_PATH)
            print("\n-> Cleanup complete: Temporary configuration file removed.")

if __name__ == "__main__":
    main()