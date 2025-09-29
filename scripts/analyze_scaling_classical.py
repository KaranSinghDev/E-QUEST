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
from scipy.optimize import curve_fit
import warnings
import inspect
from src.config_loader import load_config
from src.classical_mlp import ClassicalMLP

# ... (rest of the file is unchanged)

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


def run_empirical_benchmark(input_sizes: list, full_dataset: pd.DataFrame) -> pd.DataFrame:
    """Runs the MLP benchmark and returns a DataFrame of measured GPU time and energy."""
    results = []
    print("\n--- Starting Empirical Benchmark (Hardware-Dependent) ---")
    
    for size in input_sizes:
        print(f"\n--- Testing with input size: {size:,} ---")
        if size > len(full_dataset):
            print(f"   -> Skipping size {size:,} as it is larger than the full dataset.")
            continue
        
        subset_df = full_dataset.sample(n=size, random_state=42)
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
        
        # ADDED: Capture peak_memory_mb and accuracy_auc for saving
        results.append({
            "input_size": size,
            "measured_energy_j": measured_energy_j,
            "gpu_train_time_s": gpu_train_time_s,
            "peak_memory_mb": benchmark_results["peak_memory_mb"],
            "accuracy_auc": benchmark_results["accuracy_auc"],
            "precision": benchmark_results["precision"],
            "recall": benchmark_results["recall"]
        })
        os.remove(temp_path)
        
    return pd.DataFrame(results)

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


def find_and_validate_best_fit(results_df: pd.DataFrame):
    """Finds the best mathematical model for the data and validates its predictive power."""
    print("\n" + "="*60)
    print("🔬 FINDING AND VALIDATING THE ENERGY SCALING FUNCTION 🔬")
    print("="*60)



    validation_split_point = 10_000_000
    train_df = results_df[results_df['input_size'] <= validation_split_point]
    validation_df = results_df[results_df['input_size'] > validation_split_point]
    
    print(f"Using {len(train_df)} data points for curve fitting.")
    print(f"Using {len(validation_df)} data points for validation.")

    x_train = train_df['input_size'].values
    y_train = train_df['measured_energy_j'].values

    best_model_name, best_model_params, best_model_error = None, None, float('inf')

    print("\n--- [1] Fitting Candidate Models ---")
    for name, model_func in MODELS.items():
        # --- ADDED: Robustness Check ---
        # Get the number of parameters the model needs (e.g., a, b, c)
        num_params = len(inspect.signature(model_func).parameters) - 1 # Subtract 1 for 'x'
        
        # Only try to fit if we have enough data points
        if num_params > len(x_train):
            print(f"  - Model '{name}': Skipped (requires {num_params} data points, have {len(x_train)}).")
            continue # Go to the next model
        # --- END of ADDED Check ---
            
        try:
            params, _ = curve_fit(model_func, x_train, y_train, maxfev=100000)
            y_fit = model_func(x_train, *params)
            error = np.mean((y_train - y_fit)**2)
            print(f"  - Model '{name}': Fit successful, MSE = {error:.4f}")
            if error < best_model_error:
                best_model_error, best_model_name, best_model_params = error, name, params
        except (RuntimeError, TypeError): # Also catch TypeError for safety
            print(f"  - Model '{name}': Could not converge.")

    print(f"\n🏆 Best Fit Model for Empirical Data: '{best_model_name}' with MSE: {best_model_error:.4f}")

    if best_model_name and not validation_df.empty:
        print("\n--- [2] Validating Best Model on Unseen Data ---")
        best_func = MODELS[best_model_name]
        for _, row in validation_df.iterrows():
            x_val, y_actual = row['input_size'], row['measured_energy_j']
            y_predicted = best_func(x_val, *best_model_params)
            prediction_error_percent = (abs(y_predicted - y_actual) / y_actual) * 100
            print(f"  - For Input Size {x_val:11,}:")
            print(f"    - Actual Measured Energy:   {y_actual:8.2f} J")
            print(f"    - Predicted Energy:         {y_predicted:8.2f} J")
            print(f"    - Prediction Error:         {prediction_error_percent:.2f}%")
        print("\n✅ Validation complete. A low prediction error gives high confidence in the model.")
        
    return best_model_name, best_model_params



config = load_config()
def main():
    """Main function to drive the analysis."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
   
    print("="*60)
    print("🚀 STARTING FINAL CLASSICAL MLP SCALING ANALYSIS 🚀")
    print("="*60)
    
    try:
        full_dataset = pd.read_csv(config.ML_DATASET_PATH)
        print(f"Successfully loaded full dataset with {len(full_dataset):,} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Please run 'create_ml_dataset.py' first.")
        return
        
    empirical_results = run_empirical_benchmark(config.CLASSICAL_INPUT_SIZES, full_dataset)
    theoretical_results = run_theoretical_projection(config.CLASSICAL_INPUT_SIZES)
    final_results = pd.merge(empirical_results, theoretical_results, on="input_size")

    results_csv_path = os.path.join(config.RESULTS_DIR, "classical_results.csv")
    print(f"\n--- Saving detailed results to '{results_csv_path}' ---")
    final_results.to_csv(results_csv_path, index=False)
    print("✅ Detailed results saved.")
    
    print("\n--- Final Combined Results ---")
    print(final_results)

    best_model_name, best_model_params = find_and_validate_best_fit(empirical_results)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(final_results['input_size'], final_results['measured_energy_j'], 
            'o', label='Hardware-Dependent: Measured Energy Points (RTX 3060)', color='blue', markersize=8)
    
    ax.plot(final_results['input_size'], final_results['projected_energy_j'], 
            's', label='Hardware-Independent: Projected Energy Points (MAC Model)', color='green', markersize=8)

    if best_model_name:
        x_smooth = np.linspace(min(config.CLASSICAL_INPUT_SIZES), max(config.CLASSICAL_INPUT_SIZES), 200)
        y_smooth = MODELS[best_model_name](x_smooth, *best_model_params)
        ax.plot(x_smooth, y_smooth, '--', label=f"Best Fit Function for Measured Data: '{best_model_name}'", color='red', linewidth=2)

    ax.set_title('Energy Scaling of Classical MLP for Track Segment Classification', fontsize=16)
    ax.set_xlabel('Input Size (Number of Track Segments)', fontsize=12)
    ax.set_ylabel('Energy Consumed for Training (Joules)', fontsize=12)
    ax.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--')
    
    output_plot_path = os.path.join(config.RESULTS_DIR, "classical_mlp_energy_scaling_final.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print(f"🎉 ANALYSIS COMPLETE! Plot saved to '{output_plot_path}' 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
