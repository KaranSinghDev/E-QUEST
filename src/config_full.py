# ==============================================================================
# E-QUEST FRAMEWORK: LIVE DEMONSTRATION CONFIGURATION
# ==============================================================================
# This is a specially tuned configuration file for a live demonstration.
# It is designed to run in a reasonable time (~5-10 minutes) while
# producing scientifically meaningful and visually clear results.
# ==============================================================================

# --- 1. Global Paths ---
DATA_DIR = "data/"
RAW_EVENTS_DIR = "data/Data_Sample"
ML_DATASET_PATH = "track_segments_ml_dataset.csv"
RESULTS_DIR = "results/"

# --- 2. Data Creation Parameters ---
# Use 2 events to get a decent-sized dataset without taking too long.
NUM_EVENTS_TO_PROCESS = 2
Z_CUT = 200.0
PHI_CUT = 0.1
CHUNK_SIZE = 500

# --- 3. General Benchmarking Parameters ---
COMPUTATION_POWER_WATTS = 15.0

# --- 4. Classical MLP Analysis Parameters ---
# We use 5 data points to get a clear curve.
# The first point (50k) acts as a "warm-up" run to avoid anomalies.
# The results from this first point can be visually ignored in the final plot.
CLASSICAL_INPUT_SIZES = [50_000, 250_000, 500_000, 750_000, 1_000_000]
ENERGY_PER_MAC_JOULES = 50e-12
# We use 2 epochs to show some learning without being too slow.
MLP_EPOCHS = 2
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 4096

# --- 5. Quantum VQC Analysis Parameters ---
# We use 4 data points for the quantum curve.
QUANTUM_INPUT_SIZES = [100, 200, 300, 400]
ENERGY_PER_1Q_GATE_J = 1e-17
ENERGY_PER_2Q_GATE_J = 1e-16
# We use 2 epochs for consistency with the classical model.
VQC_EPOCHS = 2
VQC_LEARNING_RATE = 0.01
VQC_BATCH_SIZE = 128
VQC_NUM_LAYERS = 2
