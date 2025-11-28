# --- Universal Path Setup ---
import sys
import os

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
from scipy.optimize import curve_fit
import warnings
import inspect
from src.config_loader import load_config
from src.classical_mlp import ClassicalMLP


# --- [2] Define Candidate Scaling Functions for Curve Fitting ---
# We will test all requested mathematical models to see which one best fits our data.
def linear_model(x, a, b):
    return a * x + b

def logarithmic_model(x, a, b):
    # Suppress log(0) warnings if x ever starts at 0, though our data does not.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a * np.log2(x) + b

def linearithmic_model(x, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a * x * np.log2(x) + b

def polynomial_model_2(x, a, b, c): # Quadratic
    return a * x**2 + b * x + c
    
def polynomial_model_3(x, a, b, c, d): # Cubic
    return a * x**3 + b * x**2 + c * x + d

def exponential_model(x, a, b, c):
    # Use a safe exponential form to prevent overflow
    return a * np.exp(b * (x / 1_000_000)) + c # Scale x to prevent large exp values

MODELS = {
    "Linear O(n)": linear_model,
    "Logarithmic O(log n)": logarithmic_model,
    "Linearithmic O(n log n)": linearithmic_model,
    "Quadratic O(n^2)": polynomial_model_2,
    "Cubic O(n^3)": polynomial_model_3,
    "Exponential": exponential_model,
}


# --- REPLACE THE run_empirical_benchmark FUNCTION WITH THIS ---
def run_empirical_benchmark(input_sizes: list, full_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the MLP benchmark N_RUNS times for each input size to gather statistical data.
    Returns a DataFrame with raw, un-aggregated results.
    """
    raw_results = []
    print("\n--- Starting Empirical Benchmark (Hardware-Dependent) ---")
    
    # --- NEW: Outer loop for statistical runs ---
    for run_num in range(1, config.N_RUNS + 1):
        print(f"\n--- Starting Statistical Run {run_num} of {config.N_RUNS} ---")
        for size in input_sizes:
            # Use a different random seed for each run to ensure sample diversity
            current_random_state = 42 + run_num
            
            print(f"  -> Testing with input size: {size:,}")
            if size > len(full_dataset):
                print(f"     -> Skipping size {size:,} as it is larger than the full dataset.")
                continue
            
            subset_df = full_dataset.sample(n=size, random_state=current_random_state)
            temp_path = f"temp_subset_{size}.csv"
            subset_df.to_csv(temp_path, index=False)
            
            mlp_config = {
                "dataset_path": temp_path,
                "epochs": config.MLP_EPOCHS,
                "lr": config.MLP_LEARNING_RATE,
                "batch_size": config.MLP_BATCH_SIZE
            }
            mlp_algo = ClassicalMLP(config=mlp_config)
            
            benchmark_results = mlp_algo.benchmark()
            gpu_train_time_s = benchmark_results["time_training_gpu_s"]
            measured_energy_j = gpu_train_time_s * config.COMPUTATION_POWER_WATTS
            
            # Append the raw result for this specific run
            raw_results.append({
                "run_id": run_num, # Keep track of which run this was
                "input_size": size,
                "measured_energy_j": measured_energy_j,
                "gpu_train_time_s": gpu_train_time_s,
                "peak_memory_mb": benchmark_results["peak_memory_mb"],
                "accuracy_auc": benchmark_results["accuracy_auc"],
                "precision": benchmark_results["precision"],
                "recall": benchmark_results["recall"]
            })
            os.remove(temp_path)
            
    return pd.DataFrame(raw_results)

def run_theoretical_projection(input_sizes: list) -> pd.DataFrame:
    """Calculates the projected energy based on algorithmic complexity (Hardware-Independent)."""
    results = []
    print("\n--- Starting Theoretical Projection (Hardware-Independent) ---")
    macs_per_segment = (3 * 32) + (32 * 32) + (32 * 1)
    print(f"Hardware-independent complexity: {macs_per_segment} MACs per segment.")
    
    for size in input_sizes:
        total_macs_epochs = macs_per_segment * size * config.MLP_EPOCHS
        projected_energy_j = total_macs_epochs * config.ENERGY_PER_MAC_JOULES
        results.append({"input_size": size, "projected_energy_j": projected_energy_j})
        
    return pd.DataFrame(results)


# --- UPDATED FUNCTION ---
def find_and_validate_best_fit(results_df: pd.DataFrame, y_column: str):
    print("\n" + "="*60); print("🔬 FINDING AND VALIDATING THE ENERGY SCALING FUNCTION 🔬"); print("="*60)
    # ... (rest of the function logic is identical, only the input args changed)
    train_df = results_df; x_train = train_df['input_size'].values; y_train = train_df[y_column].values
    best_model_name, best_model_params, best_model_error = None, None, float('inf')
    for name, model_func in MODELS.items():
        num_params = len(inspect.signature(model_func).parameters) - 1
        if num_params > len(x_train): continue
        try:
            params, _ = curve_fit(model_func, x_train, y_train, maxfev=100000)
            y_fit = model_func(x_train, *params)
            error = np.mean((y_train - y_fit)**2)
            if error < best_model_error: best_model_error, best_model_name, best_model_params = error, name, params
        except (RuntimeError, TypeError): pass
    print(f"\n🏆 Best Fit Model for Empirical Data: '{best_model_name}' with MSE: {best_model_error:.4f}")
    return best_model_name, best_model_params

config = load_config()

# --- UPDATED main() FUNCTION ---
def main():
    """Main function to drive the analysis."""
    dataset_path = sys.argv[1]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
   
    print("="*60); print("🚀 STARTING FINAL CLASSICAL MLP SCALING ANALYSIS 🚀"); print("="*60)
    
    try:
        full_dataset = pd.read_csv(dataset_path)
        print(f"Successfully loaded full dataset with {len(full_dataset):,} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset not found at '{dataset_path}'."); return
        
    empirical_results_raw = run_empirical_benchmark(config.CLASSICAL_INPUT_SIZES, full_dataset)
    
    # --- NEW: Aggregation Step ---
    print("\n--- Aggregating Statistical Results ---")
    metrics_to_aggregate = [col for col in empirical_results_raw.columns if col not in ['run_id', 'input_size']]
    aggregated_results = empirical_results_raw.groupby('input_size')[metrics_to_aggregate].agg(['mean', 'std'])
    aggregated_results.columns = ['_'.join(col).strip() for col in aggregated_results.columns.values]
    aggregated_results.reset_index(inplace=True)
    print("✅ Aggregation complete.")
    
    theoretical_results = run_theoretical_projection(config.CLASSICAL_INPUT_SIZES)
    final_results = pd.merge(aggregated_results, theoretical_results, on="input_size")

    # Save BOTH raw and aggregated results for full transparency
    raw_csv_path = os.path.join(config.RESULTS_DIR, "classical_results_raw.csv")
    final_results_csv_path = os.path.join(config.RESULTS_DIR, "classical_results.csv")
    empirical_results_raw.to_csv(raw_csv_path, index=False)
    final_results.to_csv(final_results_csv_path, index=False)
    print(f"\n--- Saving detailed results ---"); print(f"✅ Raw results saved to '{raw_csv_path}'"); print(f"✅ Aggregated results saved to '{final_results_csv_path}'")
    
    print("\n--- Final Aggregated Results ---"); print(final_results)

    best_model_name, best_model_params = find_and_validate_best_fit(final_results, y_column='measured_energy_j_mean')
    
    # --- NEW: Plotting with Error Bands ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot the mean as a line
    ax.plot(final_results['input_size'], final_results['measured_energy_j_mean'], 'o-', label='Mean Measured Energy (Hardware-Dependent)', color='blue', markersize=8)
    # Plot the standard deviation as a shaded region
    ax.fill_between(final_results['input_size'], 
                    final_results['measured_energy_j_mean'] - final_results['measured_energy_j_std'], 
                    final_results['measured_energy_j_mean'] + final_results['measured_energy_j_std'], 
                    color='blue', alpha=0.2, label='Standard Deviation')
    
    ax.plot(final_results['input_size'], final_results['projected_energy_j'], 's--', label='Projected Energy (MAC Model)', color='green', markersize=8)

    if best_model_name:
        x_smooth = np.linspace(min(final_results['input_size']), max(final_results['input_size']), 200)
        y_smooth = MODELS[best_model_name](x_smooth, *best_model_params)
        ax.plot(x_smooth, y_smooth, '--', label=f"Best Fit Function for Mean Data: '{best_model_name}'", color='red', linewidth=2)

    ax.set_title('Energy Scaling of Classical MLP (with Statistical Error)', fontsize=16)
    # ... (rest of the plotting code is the same) ...
    ax.set_xlabel('Input Size (Number of Track Segments)', fontsize=12); ax.set_ylabel('Energy Consumed for Training (Joules)', fontsize=12)
    ax.ticklabel_format(style='plain', axis='x'); plt.xticks(rotation=45); ax.legend(fontsize=10); ax.grid(True, which='both', linestyle='--')
    
    output_plot_path = os.path.join(config.RESULTS_DIR, "classical_mlp_energy_scaling_final.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60); print(f"🎉 ANALYSIS COMPLETE! Plot saved to '{output_plot_path}' 🎉"); print("="*60)

if __name__ == "__main__":
    main()