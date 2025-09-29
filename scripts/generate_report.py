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
import matplotlib.pyplot as plt
import os
import numpy as np
from src.config_loader import load_config

# --- Plotting Style Configuration ---
C_EMPIRICAL = {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Classical (Empirical)'}
C_THEORY = {'color': 'deepskyblue', 'marker': 'v', 'linestyle': ':', 'label': 'Classical O(n) Theory'}
Q_EMPIRICAL = {'color': 'red', 'marker': 's', 'linestyle': '--', 'label': 'Quantum (Empirical)'}
Q_THEORY = {'color': 'magenta', 'marker': '^', 'linestyle': ':', 'label': 'Quantum O(n) Theory'}

def plot_energy_analysis(classical_df, quantum_df, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Analysis 1: Energy Scaling (Empirical vs. Theoretical Projection)', fontsize=18, fontweight='bold')

    # Plot 1: Classical Energy
    ax1_twin = ax1.twinx()
    ax1.plot(classical_df['input_size'], classical_df['measured_energy_j'], **C_EMPIRICAL)
    ax1_twin.plot(classical_df['input_size'], classical_df['projected_energy_j'], **C_THEORY)
    ax1.set_title('A) Classical MLP', fontsize=14); ax1.set_xlabel('Input Size'); ax1.set_ylabel('Empirical Energy on GPU (J)', color=C_EMPIRICAL['color']); ax1_twin.set_ylabel('Projected Energy (MAC Model, J)', color=C_THEORY['color']); ax1.set_yscale('log'); ax1_twin.set_yscale('log'); ax1.grid(True, which='both', linestyle='--')

    # Plot 2: Quantum Energy
    ax2_twin = ax2.twinx()
    ax2.plot(quantum_df['input_size'], quantum_df['sim_energy_j'], **Q_EMPIRICAL)
    ax2_twin.plot(quantum_df['input_size'], quantum_df['projected_energy_j'], **Q_THEORY)
    ax2.set_title('B) Quantum VQC', fontsize=14); ax2.set_xlabel('Input Size'); ax2.set_ylabel('Sim. Energy on GPU (J)', color=Q_EMPIRICAL['color']); ax2_twin.set_ylabel('Projected Energy (Gate Model, J)', color=Q_THEORY['color']); ax2.set_yscale('log'); ax2_twin.set_yscale('log'); ax2.grid(True, which='both', linestyle='--')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(save_path, "1_energy_analysis.png"), dpi=300)

def plot_memory_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(classical_df['input_size'], classical_df['peak_memory_mb'], **C_EMPIRICAL)
    ax.plot(quantum_df['input_size'], quantum_df['peak_memory_mb'], **Q_EMPIRICAL)
    ax.set_title('Analysis 2: Empirical Memory Usage\n(Theoretical Space Complexity is O(1) for both)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12); ax.legend(); ax.grid(True, linestyle='--')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "2_memory_analysis.png"), dpi=300)

def plot_performance_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(classical_df['input_size'], classical_df['accuracy_auc'], **C_EMPIRICAL)
    ax.plot(quantum_df['input_size'], quantum_df['accuracy_auc'], **Q_EMPIRICAL)
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess Baseline')
    ax.set_title('Analysis 3: Model Performance (ROC AUC)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('ROC AUC Score', fontsize=12); ax.set_ylim(0, 1); ax.legend(); ax.grid(True, linestyle='--')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "3_performance_analysis.png"), dpi=300)

def plot_time_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(classical_df['input_size'], classical_df['gpu_train_time_s'], **C_EMPIRICAL)
    ax.plot(quantum_df['input_size'], quantum_df['sim_time_gpu_s'], **Q_EMPIRICAL)
    # Scale and plot theoretical O(n) lines
    classical_const = np.mean(classical_df['gpu_train_time_s'] / classical_df['input_size'])
    quantum_const = np.mean(quantum_df['sim_time_gpu_s'] / quantum_df['input_size'])
    ax.plot(classical_df['input_size'], classical_const * classical_df['input_size'], **C_THEORY)
    ax.plot(quantum_df['input_size'], quantum_const * quantum_df['input_size'], **Q_THEORY)
    ax.set_title('Analysis 4: Execution Time\n(Theoretical Time Complexity is O(n) for both)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('Execution Time (seconds)', fontsize=12); ax.set_yscale('log'); ax.legend(); ax.grid(True, which='both', linestyle='--')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "4_time_analysis.png"), dpi=300)

# ADD THIS ENTIRE NEW FUNCTION TO YOUR SCRIPT
# REPLACE your existing plot_normalized_scaling_analysis function with this one

def plot_normalized_scaling_analysis(classical_df, quantum_df, save_path):
    """
    Generates a single, unified plot comparing the NORMALIZED scaling trends
    of all key metrics to directly compare their algorithmic complexity scaling.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Analysis 5: Normalized Cost Scaling (Trend Comparison)', fontsize=18, fontweight='bold')

    # --- THIS IS THE FIX ---
    # Create local copies of the style dictionaries without the 'label' key,
    # so we can provide more specific labels in the plot call without conflict.
    c_empirical_style = {k: v for k, v in C_EMPIRICAL.items() if k != 'label'}
    q_empirical_style = {k: v for k, v in Q_EMPIRICAL.items() if k != 'label'}
    c_theory_style = {k: v for k, v in C_THEORY.items() if k != 'label'}
    q_theory_style = {k: v for k, v in Q_THEORY.items() if k != 'label'}
    # --- END OF FIX ---

    # --- Normalization Logic ---
    epsilon = 1e-12
    norm_c_time = classical_df['gpu_train_time_s'] / (classical_df['gpu_train_time_s'].iloc[0] + epsilon)
    norm_q_time = quantum_df['sim_time_gpu_s'] / (quantum_df['sim_time_gpu_s'].iloc[0] + epsilon)
    norm_c_theory = classical_df['projected_energy_j'] / (classical_df['projected_energy_j'].iloc[0] + epsilon)
    norm_q_theory = quantum_df['projected_energy_j'] / (quantum_df['projected_energy_j'].iloc[0] + epsilon)
    
    # --- Plotting ---
    # Now we use the local style dicts and provide the new labels.
    ax.plot(classical_df['input_size'], norm_c_time, **c_empirical_style, label='Classical Time Scaling')
    ax.plot(quantum_df['input_size'], norm_q_time, **q_empirical_style, label='Quantum Time Scaling')
    ax.plot(classical_df['input_size'], norm_c_theory, **c_theory_style, label='Classical Projected Energy Scaling')
    ax.plot(quantum_df['input_size'], norm_q_theory, **q_theory_style, label='Quantum Projected Energy Scaling')

    ax.set_title('Scaling Factor vs. Input Size', fontsize=14)
    ax.set_xlabel('Input Size', fontsize=12)
    ax.set_ylabel('Normalized Cost (Scaling Factor)', fontsize=12)
    ax.axhline(y=1.0, color='gray', linestyle=':', label='Baseline')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "5_normalized_scaling_analysis.png"), dpi=300)

config = load_config()

def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("="*60); print("📈 GENERATING ADVANCED SCIENTIFIC REPORTS 📈"); print("="*60)
    try:
        classical_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "classical_results.csv"))
        quantum_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "quantum_results.csv"))
        print("✅ Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a results file: {e}\nPlease run analysis scripts first."); return

    print("-> Generating 4 individual analysis plots...")
    plot_energy_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_memory_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_performance_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_time_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_normalized_scaling_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plt.close('all') # Close plots to free up memory
    print(f"✅ All 4 report plots saved successfully to '{config.RESULTS_DIR}'")
    print("="*60)

if __name__ == "__main__":
    main()
