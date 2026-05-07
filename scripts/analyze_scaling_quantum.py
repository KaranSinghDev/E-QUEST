# --- Universal Path Setup ---
import sys
import os
# This block of code is designed to solve the ModuleNotFoundError
# by dynamically adding the project's root directory to the Python path.
# This allows the script to be run from anywhere, either directly or as a subprocess.
try:
    # Get the absolute path of the directory containing the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project's root directory.
    project_root = os.path.dirname(script_dir)
    # Add the project root to the system path if it's not already there.
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    # This fallback is for interactive environments where __file__ might not be defined.
    # It assumes the current working directory is the project root.
    if '.' not in sys.path:
        sys.path.append('.')
# --- End of Universal Path Setup ---



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from src.config_loader import load_config
from src.quantum_vqc import QuantumVQC



# --- REPLACE THE ENTIRE run_quantum_benchmark FUNCTION WITH THIS ---
def run_quantum_benchmark(input_sizes: list, full_dataset: pd.DataFrame, config) -> pd.DataFrame:
    """
    Runs the QuantumVQC benchmark N_RUNS times for each data subset size to 
    gather statistical data.
    """
    raw_results = []
    print("\n--- Starting Quantum VQC Empirical Benchmark ---")
    
    # --- NEW: Outer loop for statistical runs ---
    for run_num in range(1, config.N_RUNS + 1):
        print(f"\n--- Starting Statistical Run {run_num} of {config.N_RUNS} ---")
        for size in input_sizes:
            # Use a different random seed for each run for both data sampling and model initialization
            current_random_state = 42 + run_num
            torch.manual_seed(current_random_state)

            print(f"  -> Testing with input size: {size:,}")
            if size > len(full_dataset):
                print(f"     -> Skipping size {size:,} as it is larger than the full dataset.")
                continue
            
            # Configure and run the benchmark
            vqc_config = {
                "dataset": full_dataset, # Pass the entire DataFrame
                "num_samples": size,
                "epochs": config.VQC_EPOCHS,
                "num_layers": config.VQC_NUM_LAYERS,
                "lr": config.VQC_LEARNING_RATE,
                "batch_size": config.VQC_BATCH_SIZE
            }
            
            q_algo = QuantumVQC(config=vqc_config)
            
            benchmark_results = q_algo.benchmark()
                    
            # Add the run_id to the results dictionary for aggregation
            result_row = {
                "run_id": run_num,
                "input_size": size,
                "real_energy_j": benchmark_results.get("real_energy_j",
                                  benchmark_results.get("sim_time_gpu_s", 0) * 15.0),
                "energy_source": benchmark_results.get("energy_source", "estimated"),
                "zeus_window_s": benchmark_results.get("zeus_window_s",
                                  benchmark_results.get("sim_time_gpu_s", 0)),
            }
            result_row.update(benchmark_results)
            raw_results.append(result_row)
            
    return pd.DataFrame(raw_results)

config = load_config()
# --- REPLACE THE ENTIRE main FUNCTION WITH THIS ---
def main():
    """Main function to drive the quantum scaling analysis and generate the final plot."""
    dataset_path = sys.argv[1]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("="*60)
    print("🚀 STARTING QUANTUM VQC ENERGY SCALING ANALYSIS 🚀")
    print("="*60)
    
    try:
        full_dataset = pd.read_csv(dataset_path)
        print(f"Successfully loaded full dataset with {len(full_dataset):,} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset not found at '{dataset_path}'."); return
        
    benchmark_results_raw = run_quantum_benchmark(config.QUANTUM_INPUT_SIZES, full_dataset, config)
    
    if benchmark_results_raw.empty:
        print("\n❌ No benchmark data was generated. Aborting analysis.")
        return

    # --- NEW: Aggregation Step ---
    print("\n--- Aggregating Statistical Results ---")
    metrics_to_aggregate = [col for col in benchmark_results_raw.columns if col not in ['run_id', 'input_size']]
    aggregated_results = benchmark_results_raw.groupby('input_size')[metrics_to_aggregate].agg(['mean', 'std'])
    aggregated_results.columns = ['_'.join(col).strip() for col in aggregated_results.columns.values]
    aggregated_results.reset_index(inplace=True)
    print("✅ Aggregation complete.")
        
    # --- [3] Calculate the Energy Curves from the MEAN values ---
    print("\n--- Calculating Energy Scaling Curves ---")

    aggregated_results['sim_energy_j_mean'] = aggregated_results['sim_time_gpu_s_mean'] * config.COMPUTATION_POWER_WATTS
    aggregated_results['sim_energy_j_std'] = aggregated_results['sim_time_gpu_s_std'] * config.COMPUTATION_POWER_WATTS

    total_1q_gates_mean = aggregated_results['n_1q_gates_mean'] * aggregated_results['total_calls_mean']
    total_2q_gates_mean = aggregated_results['n_2q_gates_mean'] * aggregated_results['total_calls_mean']
    
    aggregated_results['projected_energy_j_mean'] = (total_1q_gates_mean * config.ENERGY_PER_1Q_GATE_J) + (total_2q_gates_mean * config.ENERGY_PER_2Q_GATE_J)
    # We can ignore the std deviation for the projected energy as gate counts are deterministic.
    aggregated_results['projected_energy_j_std'] = 0 
    print("✅ Calculated energy scaling curves.")
    
    print("\n--- Final Aggregated Results ---")
    print(aggregated_results)

    # Save both raw and aggregated results
    raw_csv_path = os.path.join(config.RESULTS_DIR, "quantum_results_raw.csv")
    final_csv_path = os.path.join(config.RESULTS_DIR, "quantum_results.csv")
    benchmark_results_raw.to_csv(raw_csv_path, index=False)
    aggregated_results.to_csv(final_csv_path, index=False)
    print(f"\n--- Saving detailed results ---"); print(f"✅ Raw results saved to '{raw_csv_path}'"); print(f"✅ Aggregated results saved to '{final_csv_path}'")

    # --- NEW: Generate the Final Plot with Error Bands ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Mean Simulation Energy
    ax.plot(aggregated_results['input_size'], aggregated_results['sim_energy_j_mean'], 'o-', label=f'Mean Sim. Energy on GPU ({config.COMPUTATION_POWER_WATTS}W)', color='red', linewidth=2)
    # Plot Simulation Energy Standard Deviation
    ax.fill_between(aggregated_results['input_size'],
                    aggregated_results['sim_energy_j_mean'] - aggregated_results['sim_energy_j_std'],
                    aggregated_results['sim_energy_j_mean'] + aggregated_results['sim_energy_j_std'],
                    color='red', alpha=0.2, label='Standard Deviation (Sim. Energy)')

    ax2 = ax.twinx()
    # Plot Mean Projected Energy
    ax2.plot(aggregated_results['input_size'], aggregated_results['projected_energy_j_mean'], 's--', label='Mean Projected Energy (Gate Model)', color='green', linewidth=2)

    ax.set_title('Energy Scaling of Quantum VQC (with Statistical Error)', fontsize=16)
    ax.set_xlabel('Input Size (Number of Training Samples)', fontsize=12)
    ax.set_ylabel('Energy for Simulation on GPU (Joules)', fontsize=12, color='red')
    ax2.set_ylabel('Projected Energy on Quantum Hardware (Joules)', fontsize=12, color='green')
    
    ax.set_yscale('log'); ax2.set_yscale('log')
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax.grid(True, which='both', linestyle='--')
    
    output_plot_path = os.path.join(config.RESULTS_DIR, "quantum_vqc_energy_scaling.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print(f"🎉 ANALYSIS COMPLETE! Plot saved to '{output_plot_path}' 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
