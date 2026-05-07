# --- Universal Path Setup ---
import sys
import os
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    if '.' not in sys.path:
        sys.path.append('.')
# --- End of Universal Path Setup ---

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from src.config_loader import load_config

# --- Plotting Style Configuration (Unchanged) ---
C_EMPIRICAL_MEAN = {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Classical Mean'}
C_THEORY = {'color': 'deepskyblue', 'marker': 'v', 'linestyle': ':', 'label': 'Classical Theory'}
Q_EMPIRICAL_MEAN = {'color': 'red', 'marker': 's', 'linestyle': '--', 'label': 'Quantum Mean'}
Q_THEORY = {'color': 'magenta', 'marker': '^', 'linestyle': ':', 'label': 'Quantum Theory'}

# --- UPDATED PLOTTING FUNCTIONS ---

def plot_energy_analysis(classical_df, quantum_df, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Analysis 1: Energy Scaling (with Statistical Error)', fontsize=18, fontweight='bold')

    # --- Plot 1: Classical Energy ---
    # Legacy flat-constant estimate (time × fixed watts)
    ax1.plot(classical_df['input_size'], classical_df['measured_energy_j_mean'],
             color='steelblue', marker='o', linestyle='--',
             label='Flat-Constant Estimate (time × 15W)')
    ax1.fill_between(classical_df['input_size'],
                     classical_df['measured_energy_j_mean'] - classical_df['measured_energy_j_std'],
                     classical_df['measured_energy_j_mean'] + classical_df['measured_energy_j_std'],
                     color='steelblue', alpha=0.15)
    # Real Zeus measurement (if available in CSV)
    if 'real_energy_j_mean' in classical_df.columns:
        ax1.plot(classical_df['input_size'], classical_df['real_energy_j_mean'],
                 **C_EMPIRICAL_MEAN)
        ax1.fill_between(classical_df['input_size'],
                         classical_df['real_energy_j_mean'] - classical_df['real_energy_j_std'],
                         classical_df['real_energy_j_mean'] + classical_df['real_energy_j_std'],
                         color=C_EMPIRICAL_MEAN['color'], alpha=0.2, label='Zeus Real Energy ± Std. Dev.')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(classical_df['input_size'], classical_df['projected_energy_j'], **C_THEORY)
    ax1.set_title('A) Classical MLP', fontsize=14)
    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Measured Energy on GPU (J)', color=C_EMPIRICAL_MEAN['color'])
    ax1_twin.set_ylabel('Projected Energy (MAC Model, J)', color=C_THEORY['color'])
    ax1.set_yscale('log'); ax1_twin.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend(loc='upper left', fontsize=8)

    # --- Plot 2: Quantum Energy ---
    ax2.plot(quantum_df['input_size'], quantum_df['sim_energy_j_mean'],
             color='tomato', marker='o', linestyle='--',
             label='Flat-Constant Estimate (time × 15W)')
    ax2.fill_between(quantum_df['input_size'],
                     quantum_df['sim_energy_j_mean'] - quantum_df['sim_energy_j_std'],
                     quantum_df['sim_energy_j_mean'] + quantum_df['sim_energy_j_std'],
                     color='tomato', alpha=0.15)
    if 'real_energy_j_mean' in quantum_df.columns:
        ax2.plot(quantum_df['input_size'], quantum_df['real_energy_j_mean'],
                 **Q_EMPIRICAL_MEAN)
        ax2.fill_between(quantum_df['input_size'],
                         quantum_df['real_energy_j_mean'] - quantum_df['real_energy_j_std'],
                         quantum_df['real_energy_j_mean'] + quantum_df['real_energy_j_std'],
                         color=Q_EMPIRICAL_MEAN['color'], alpha=0.2, label='Zeus Real Energy ± Std. Dev.')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(quantum_df['input_size'], quantum_df['projected_energy_j_mean'], **Q_THEORY)
    ax2.set_title('B) Quantum VQC', fontsize=14)
    ax2.set_xlabel('Input Size')
    ax2.set_ylabel('Measured GPU Energy (J)', color=Q_EMPIRICAL_MEAN['color'])
    ax2_twin.set_ylabel('Projected Energy (Gate Model, J)', color=Q_THEORY['color'])
    ax2.set_yscale('log'); ax2_twin.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "1_energy_analysis.png"), dpi=300)

def plot_memory_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(classical_df['input_size'], classical_df['peak_memory_mb_mean'], **C_EMPIRICAL_MEAN)
    ax.fill_between(classical_df['input_size'],
                     classical_df['peak_memory_mb_mean'] - classical_df['peak_memory_mb_std'],
                     classical_df['peak_memory_mb_mean'] + classical_df['peak_memory_mb_std'],
                     color=C_EMPIRICAL_MEAN['color'], alpha=0.2)
    ax.plot(quantum_df['input_size'], quantum_df['peak_memory_mb_mean'], **Q_EMPIRICAL_MEAN)
    ax.fill_between(quantum_df['input_size'],
                     quantum_df['peak_memory_mb_mean'] - quantum_df['peak_memory_mb_std'],
                     quantum_df['peak_memory_mb_mean'] + quantum_df['peak_memory_mb_std'],
                     color=Q_EMPIRICAL_MEAN['color'], alpha=0.2)
    ax.set_title('Analysis 2: Mean Empirical Memory Usage', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12); ax.legend(); ax.grid(True, linestyle='--')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "2_memory_analysis.png"), dpi=300)

def plot_performance_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(classical_df['input_size'], classical_df['accuracy_auc_mean'], **C_EMPIRICAL_MEAN)
    ax.fill_between(classical_df['input_size'],
                     classical_df['accuracy_auc_mean'] - classical_df['accuracy_auc_std'],
                     classical_df['accuracy_auc_mean'] + classical_df['accuracy_auc_std'],
                     color=C_EMPIRICAL_MEAN['color'], alpha=0.2)
    ax.plot(quantum_df['input_size'], quantum_df['accuracy_auc_mean'], **Q_EMPIRICAL_MEAN)
    ax.fill_between(quantum_df['input_size'],
                     quantum_df['accuracy_auc_mean'] - quantum_df['accuracy_auc_std'],
                     quantum_df['accuracy_auc_mean'] + quantum_df['accuracy_auc_std'],
                     color=Q_EMPIRICAL_MEAN['color'], alpha=0.2)
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess Baseline')
    ax.set_title('Analysis 3: Mean Model Performance (ROC AUC)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('ROC AUC Score', fontsize=12); ax.set_ylim(0, 1); ax.legend(); ax.grid(True, linestyle='--')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "3_performance_analysis.png"), dpi=300)

def plot_time_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(classical_df['input_size'], classical_df['gpu_train_time_s_mean'], **C_EMPIRICAL_MEAN)
    ax.fill_between(classical_df['input_size'],
                     classical_df['gpu_train_time_s_mean'] - classical_df['gpu_train_time_s_std'],
                     classical_df['gpu_train_time_s_mean'] + classical_df['gpu_train_time_s_std'],
                     color=C_EMPIRICAL_MEAN['color'], alpha=0.2)
    ax.plot(quantum_df['input_size'], quantum_df['sim_time_gpu_s_mean'], **Q_EMPIRICAL_MEAN)
    ax.fill_between(quantum_df['input_size'],
                     quantum_df['sim_time_gpu_s_mean'] - quantum_df['sim_time_gpu_s_std'],
                     quantum_df['sim_time_gpu_s_mean'] + quantum_df['sim_time_gpu_s_std'],
                     color=Q_EMPIRICAL_MEAN['color'], alpha=0.2)
    ax.set_title('Analysis 4: Mean Execution Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('Execution Time (seconds)', fontsize=12); ax.set_yscale('log'); ax.legend(); ax.grid(True, which='both', linestyle='--')
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "4_time_analysis.png"), dpi=300)

# The normalized plot does not need error bands, as it's for trend comparison
def plot_normalized_scaling_analysis(classical_df, quantum_df, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Analysis 5: Normalized Cost Scaling (Trend Comparison)', fontsize=18, fontweight='bold')
    epsilon = 1e-12
    norm_c_time = classical_df['gpu_train_time_s_mean'] / (classical_df['gpu_train_time_s_mean'].iloc[0] + epsilon)
    norm_q_time = quantum_df['sim_time_gpu_s_mean'] / (quantum_df['sim_time_gpu_s_mean'].iloc[0] + epsilon)
    norm_c_theory = classical_df['projected_energy_j'] / (classical_df['projected_energy_j'].iloc[0] + epsilon)
    norm_q_theory = quantum_df['projected_energy_j_mean'] / (quantum_df['projected_energy_j_mean'].iloc[0] + epsilon)
    
    ax.plot(classical_df['input_size'], norm_c_time, 'o-', color=C_EMPIRICAL_MEAN['color'], label='Classical Time Scaling')
    ax.plot(quantum_df['input_size'], norm_q_time, 's--', color=Q_EMPIRICAL_MEAN['color'], label='Quantum Time Scaling')
    ax.plot(classical_df['input_size'], norm_c_theory, 'v:', color=C_THEORY['color'], label='Classical Projected Energy Scaling')
    ax.plot(quantum_df['input_size'], norm_q_theory, '^:', color=Q_THEORY['color'], label='Quantum Projected Energy Scaling')

    ax.set_title('Scaling Factor vs. Input Size', fontsize=14)
    ax.set_xlabel('Input Size', fontsize=12); ax.set_ylabel('Normalized Cost (Scaling Factor)', fontsize=12)
    ax.axhline(y=1.0, color='gray', linestyle=':', label='Baseline'); ax.legend(); ax.grid(True, which='both', linestyle='--'); ax.set_yscale('log')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(save_path, "5_normalized_scaling_analysis.png"), dpi=300)

config = load_config()

def main():
    # The conductor now passes the final balanced dataset path
    dataset_path = sys.argv[1] # This script doesn't use it, but we accept it for consistency
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("="*60); print("📈 GENERATING ADVANCED SCIENTIFIC REPORTS 📈"); print("="*60)
    try:
        # Load the AGGREGATED results files
        classical_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "classical_results.csv"))
        quantum_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "quantum_results.csv"))
        print("✅ Aggregated data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a results file: {e}\nPlease run analysis scripts first."); return

    print("-> Generating 5 individual analysis plots (now with error bands)...")
    plot_energy_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_memory_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_performance_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_time_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plot_normalized_scaling_analysis(classical_df, quantum_df, config.RESULTS_DIR)
    plt.close('all')
    print(f"✅ All 5 report plots saved successfully to '{config.RESULTS_DIR}'")
    
    # Also update the final markdown report (if generate_report is not a separate script anymore)
    # This logic would typically be in the main run_benchmark.py
    
    print("="*60)

if __name__ == "__main__":
    main()