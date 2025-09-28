# ==============================================================================
# E-QUEST FRAMEWORK: SMOKE TEST CONFIGURATION
# ==============================================================================
# This is a lightweight configuration file designed to run the entire framework
# quickly to verify that all components are working correctly.
# ==============================================================================

# --- 1. Global Paths ---
DATA_DIR = "data/"
RAW_EVENTS_DIR = f"{DATA_DIR}/train_100_events"
ML_DATASET_PATH = "track_segments_ml_dataset.csv"
RESULTS_DIR = "results/"

# --- 2. Data Creation Parameters (Fast) ---
NUM_EVENTS_TO_PROCESS = 2   # Only process one event.
Z_CUT = 200.0
PHI_CUT = 0.1
CHUNK_SIZE = 500

# --- 3. General Benchmarking Parameters ---
COMPUTATION_POWER_WATTS = 15.0

# --- 4. Classical MLP Analysis Parameters (Fast) ---
CLASSICAL_INPUT_SIZES = [10_000, 20_000] # Use only two small sizes.
ENERGY_PER_MAC_JOULES = 50e-12
MLP_EPOCHS = 1 # Only one epoch.
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 4096

# --- 5. Quantum VQC Analysis Parameters (Fast) ---
QUANTUM_INPUT_SIZES = [50, 100] # Use only two very small sizes.
ENERGY_PER_1Q_GATE_J = 1e-17
ENERGY_PER_2Q_GATE_J = 1e-16
VQC_EPOCHS = 1 # Only one epoch.
VQC_LEARNING_RATE = 0.01
VQC_BATCH_SIZE = 128
VQC_NUM_LAYERS = 2