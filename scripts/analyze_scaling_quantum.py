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



def run_quantum_benchmark(input_sizes: list, full_dataset: pd.DataFrame, config) -> pd.DataFrame:
    """
    Runs the QuantumVQC benchmark on different data subset sizes and returns the results,
    including both hardware-dependent time and hardware-independent gate counts.
    """
    results = []
    print("\n--- Starting Quantum VQC Empirical Benchmark ---")
    
    for size in input_sizes:
        print(f"\n--- Testing with input size: {size:,} ---")
        if size > len(full_dataset):
            print(f"   -> Skipping size {size:,} as it is larger than the full dataset.")
            continue
        
        # We need a temporary file for the algorithm to read its balanced subset from.
        temp_path = f"temp_subset_{size}.csv"
        # We sample a larger amount to ensure we have enough true/false segments
        full_dataset.sample(n=min(size*10, len(full_dataset)), random_state=42).to_csv(temp_path, index=False)
        
        # Configure and run the benchmark
        vqc_config = {
            "dataset_path": temp_path,
            "num_samples": size,
            "epochs": config.VQC_EPOCHS,
            "num_layers": config.VQC_NUM_LAYERS,
            "lr": config.VQC_LEARNING_RATE,
            "batch_size": config.VQC_BATCH_SIZE
        }
        
        # Set a seed for reproducibility
        torch.manual_seed(42)
        q_algo = QuantumVQC(config=vqc_config)
        
        benchmark_results = q_algo.benchmark()
                
        results.append({
            "input_size": size,
            "sim_time_gpu_s": benchmark_results["time_training_gpu_s"],
            "accuracy_auc": benchmark_results["accuracy_auc"],
            "precision": benchmark_results["precision"],
            "recall": benchmark_results["recall"],
            "peak_memory_mb": benchmark_results["peak_memory_mb"],
            "n_1q_gates": benchmark_results["1q_gates_per_call"],
            "n_2q_gates": benchmark_results["2q_gates_per_call"],
            "circuit_depth": benchmark_results["circuit_depth"],
            "total_calls": benchmark_results["total_training_calls"]
        })
                
        os.remove(temp_path)
        
    return pd.DataFrame(results)

config = load_config()
def main():
    """Main function to drive the quantum scaling analysis and generate the final plot."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
     # ADD THIS LINE
    print("="*60)
    print("🚀 STARTING QUANTUM VQC ENERGY SCALING ANALYSIS 🚀")
    print("="*60)
    
    try:
        full_dataset = pd.read_csv(config.ML_DATASET_PATH)
        print(f"Successfully loaded full dataset with {len(full_dataset):,} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Please run 'create_ml_dataset.py' first.")
        return
        
    benchmark_results  = run_quantum_benchmark(config.QUANTUM_INPUT_SIZES, full_dataset, config)
    
    if benchmark_results.empty:
        print("\n❌ No benchmark data was generated. Aborting analysis.")
        return
        
    # --- [3] Calculate the Energy Curves ---
    print("\n--- Calculating Energy Scaling Curves ---")

    # Curve 1: Hardware-Dependent Simulation Cost
    benchmark_results['sim_energy_j'] = benchmark_results['sim_time_gpu_s'] * config.COMPUTATION_POWER_WATTS
    print("✅ Calculated hardware-dependent simulation energy.")

    # Curve 2: Hardware-Independent Projected Cost
    # Total gates = (gates per call) * (number of calls during training)
    total_1q_gates = benchmark_results['n_1q_gates'] * benchmark_results['total_calls']
    total_2q_gates = benchmark_results['n_2q_gates'] * benchmark_results['total_calls']
    
    benchmark_results['projected_energy_j'] = (total_1q_gates * config.ENERGY_PER_1Q_GATE_J) + \
                                              (total_2q_gates * config.ENERGY_PER_2Q_GATE_J)
    print("✅ Calculated hardware-independent projected energy.")
    
    print("\n--- Final Combined Results ---")
    print(benchmark_results)

    results_csv_path = os.path.join(config.RESULTS_DIR, "quantum_results.csv")
    print(f"\n--- Saving detailed results to '{results_csv_path}' ---")
    benchmark_results.to_csv(results_csv_path, index=False)
    print("✅ Detailed results saved.")

    # --- Generate the Final Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the hardware-dependent, empirical simulation results
    ax.plot(benchmark_results['input_size'], benchmark_results['sim_energy_j'], 
            'o-', label=f'Hardware-Dependent: Simulation Energy on RTX 3060 ({config.COMPUTATION_POWER_WATTS}W)', color='red', linewidth=2)
            
    # Plot the hardware-independent, theoretical projection
    # NOTE: We create a second y-axis because the projected energy is many orders of magnitude smaller.
    ax2 = ax.twinx()
    ax2.plot(benchmark_results['input_size'], benchmark_results['projected_energy_j'], 
            's--', label='Hardware-Independent: Projected Energy (Gate-Based Model)', color='green', linewidth=2)

    ax.set_title('Energy Scaling of Quantum VQC for Track Segment Classification', fontsize=16)
    ax.set_xlabel('Input Size (Number of Training Samples)', fontsize=12)
    ax.set_ylabel('Energy for Simulation on GPU (Joules)', fontsize=12, color='red')
    ax2.set_ylabel('Projected Energy on Quantum Hardware (Joules)', fontsize=12, color='green')
    
    # Use logarithmic scale to handle the vast difference in energy scales
    ax.set_yscale('log')
    ax2.set_yscale('log')
    
    # Ask matplotlib to combine the legends from both aaxis
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
