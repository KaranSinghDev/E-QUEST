import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import time
from pennylane import numpy as np
import pennylane as qml
from src.base_algorithm import BaseAlgorithm

logging.getLogger("zeus").setLevel(logging.WARNING)

try:
    from zeus.monitor import ZeusMonitor as _ZeusMonitor
    _ZEUS_AVAILABLE = True
except ImportError:
    _ZEUS_AVAILABLE = False

# --- Part 1: Define the Quantum Circuit (QNode) ---

# Global variables for the quantum device and number of qubits
# We define them here so they can be set once and used by all functions.
NUM_QUBITS = 4 # Using 4 qubits is a good starting point for a VQC
DEV = None # This will be our quantum device

def create_quantum_device(wires: int):
    """
    Tries to initialize the high-performance GPU device, with a fallback to CPU.
    """
    global DEV
    print("  -> Attempting to initialize quantum device...")
    try:
        # Try to get the lightning.gpu device for fast simulation
        DEV = qml.device("lightning.gpu", wires=wires)
        print(f"✅ SUCCESS: PennyLane is using 'lightning.gpu' with {wires} qubits.")
    except qml.DeviceError:
        print("⚠️ WARNING: 'lightning.gpu' not available. Falling back to 'default.qubit' (CPU).")
        DEV = qml.device("default.qubit", wires=wires)
    print(f"     Device selected: {DEV.name}")


# Create the device when the script is loaded
create_quantum_device(wires=NUM_QUBITS)


@qml.qnode(DEV, interface='torch', diff_method='parameter-shift')
def vqc_circuit(inputs, weights):
    """
    The core Variational Quantum Classifier circuit.
    
    Args:
        inputs (torch.Tensor): A tensor of our 3 classical features.
        weights (torch.Tensor): The trainable parameters (weights) for the quantum gates.
    """
    # 1. Data Embedding: Encode the 3 classical features into the quantum state.
    qml.templates.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    
    # 2. Variational Layers: The "learnable" part of the circuit.
    # We use a standard template for a powerful variational circuit.
    qml.templates.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    
    # 3. Measurement: Return the expectation value of a single qubit.
    # This gives a classical output between -1 and 1.
    return qml.expval(qml.PauliZ(wires=0))


# --- Part 2: The Main Algorithm Wrapper Class ---
class QuantumVQC(BaseAlgorithm):
    """
    A wrapper for our Variational Quantum Classifier. This class handles data loading,
    training, and benchmarking, conforming to the BaseAlgorithm interface.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        
        # --- Model and Training Configuration ---
        self.dataset = self.config.get("dataset") # Get the DataFrame directly
        self.epochs = self.config.get("epochs", 3)
        self.learning_rate = self.config.get("lr", 0.01)
        self.batch_size = self.config.get("batch_size", 128)
        
        # --- Define the shape of our trainable weights ---
        num_layers = self.config.get("num_layers", 2)
        self.weights_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=NUM_QUBITS)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  -> PyTorch components (optimizer) will use device: {self.device}")
        

        self.weights = torch.randn(self.weights_shape, device=self.device, dtype=torch.float64, requires_grad=True)

# --- REPLACE THE ENTIRE _load_and_prepare_data METHOD WITH THIS ---
    def _load_and_prepare_data(self):
        """
        Loads and prepares the (now pre-balanced) dataset for the VQC.
        This method assumes it is receiving a DataFrame that has already been
        globally balanced by the 'preprocess_ml_dataset.py' script.
        """
        print("  -> Preparing ML data from in-memory DataFrame...")
        df = self.dataset # Use the DataFrame passed during initialization

        # The 'num_samples' config now specifies how many samples to draw
        # from the pre-balanced dataset for this specific benchmark run.
        num_samples = self.config.get("num_samples")
        
        # Sample the required number of rows from the balanced dataset.
        if num_samples > len(df):
            print(f"⚠️ WARNING: Requested {num_samples} samples, but balanced dataset only has {len(df)}. Using all available samples.")
            subset_df = df
        else:
            subset_df = df.sample(n=num_samples, random_state=42)
        
        X = subset_df[['delta_r', 'delta_phi', 'delta_z']].values
        y = subset_df['label'].values
        
        # Rescale y from {0, 1} to {-1, 1} for the VQC's output format
        y = y * 2 - 1 
        
        # Perform the train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Convert to tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float64)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float64).view(-1, 1)
        
        print(f"  -> Data ready. Training samples: {len(self.X_train_tensor)}, Validation samples: {len(self.X_val_tensor)}")

    def train(self) -> float:
        """
        Trains the VQC and returns the total computation time in seconds, using the
        appropriate timer for the hardware being used (GPU or CPU).
        """
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam([self.weights], lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # --- Hardware-Aware Timer Setup ---
        # Check if we are actually using the GPU backend for timing.
        use_gpu_timer = "gpu" in DEV.name and torch.cuda.is_available()
        
        if use_gpu_timer:
            print("   -> Using high-precision GPU timer (torch.cuda.Event).")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        else:
            print("   -> Using high-precision CPU timer (time.perf_counter).")
        
        total_computation_time_s = 0

        print(f"  -> Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_wall_time_start = time.time()
            for features, labels in train_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # --- Timing Logic ---
                if use_gpu_timer:
                    start_event.record()
                else:
                    cpu_start_time = time.perf_counter()

                optimizer.zero_grad()
                predictions = torch.stack([vqc_circuit(f, self.weights) for f in features]).to(torch.float64).view(-1, 1)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                if use_gpu_timer:
                    end_event.record()
                    torch.cuda.synchronize()
                    total_computation_time_s += start_event.elapsed_time(end_event) / 1000.0 # Convert ms to s
                else:
                    cpu_end_time = time.perf_counter()
                    total_computation_time_s += cpu_end_time - cpu_start_time

            print(f"     Epoch [{epoch+1}/{self.epochs}], Wall Clock Time: {time.time() - epoch_wall_time_start:.2f}s")
                
        return total_computation_time_s


    def evaluate(self) -> dict:
        """
        Evaluates the VQC on the validation set and returns a dictionary of
        performance metrics: AUC, Precision, and Recall.
        """
        with torch.no_grad():
            X_val_device = self.X_val_tensor.to(self.device)
            # Get the raw predictions from the model, which are in the range [-1, 1]
            predictions_raw = torch.stack([vqc_circuit(f, self.weights) for f in X_val_device]).cpu().numpy()
            
            # Get the ground truth labels, which are also in {-1, 1}
            true_labels_raw = self.y_val_tensor.numpy()


        # 1. For AUC, convert both to probabilities {0, 1}
        probs_for_auc = (predictions_raw + 1) / 2
        true_labels_for_auc = (true_labels_raw + 1) / 2
        auc_score = roc_auc_score(true_labels_for_auc, probs_for_auc)
        
        # 2. For Precision/Recall, convert predictions to binary {-1, 1}
        # A positive raw prediction (> 0) corresponds to a prediction of class 1.
        predictions_binary = np.sign(predictions_raw)
        # We need to handle the case where a prediction is exactly 0.
        # Let's assign it to the negative class (-1) for consistency.
        predictions_binary[predictions_binary == 0] = -1

        # We need to ensure scikit-learn uses the correct labels {1, -1}
        # The `pos_label=1` argument is crucial here.
        precision = precision_score(true_labels_raw, predictions_binary, pos_label=1, zero_division=0)
        recall = recall_score(true_labels_raw, predictions_binary, pos_label=1, zero_division=0)


        # Return all metrics in a dictionary
        return {
            "accuracy_auc": auc_score,
            "precision": precision,
            "recall": recall
        }

    def get_gate_counts(self) -> dict:
        """
        Calculates hardware-independent gate counts by decomposing the circuit.
        This method is confirmed to be robust from our test scripts.
        """
        one_input_sample = self.X_train_tensor[0].cpu()
        cpu_weights = self.weights.cpu()

        with qml.tape.QuantumTape() as tape:
            vqc_circuit(one_input_sample.detach().numpy(), cpu_weights.detach().numpy())
            
        [expanded_tape], _ = qml.devices.preprocess.decompose(
            tape, 
            stopping_condition=lambda op: op.num_wires is not None and op.num_wires <= 2
        )
        
        total_1q_gates = sum(1 for op in expanded_tape.operations if op.num_wires == 1)
        total_2q_gates = sum(1 for op in expanded_tape.operations if op.num_wires == 2)
        
        return {"n_1q_gates": total_1q_gates, "n_2q_gates": total_2q_gates}


    def get_circuit_specs(self) -> dict:
        """
        Calculates the circuit depth using the robust, manual method
        verified in our exploration scripts.
        """
        with qml.tape.QuantumTape() as tape:
            vqc_circuit(self.X_train_tensor[0], self.weights)
        [decomposed_tape], _ = qml.devices.preprocess.decompose(
            tape,
            stopping_condition=lambda op: op.num_wires is not None and op.num_wires <= 2
        )
        
        wire_depths = np.zeros(NUM_QUBITS, dtype=int)
        for op in decomposed_tape.operations:
            op_wires = list(op.wires)
            max_prev_depth = 0
            if op_wires:
                max_prev_depth = np.max(wire_depths[op_wires])
            new_depth = max_prev_depth + 1
            for wire_idx in op_wires:
                wire_depths[wire_idx] = new_depth
        
        circuit_depth = np.max(wire_depths) if len(wire_depths) > 0 else 0
        return {"circuit_depth": circuit_depth}


    def benchmark(self) -> dict:
        """
        Orchestrates the full benchmark for the VQC, measuring time, real GPU
        energy (via ZeusMonitor), memory, gate counts, circuit depth, and
        performance metrics.
        """
        self._load_and_prepare_data()

        # --- Feature 1: Real GPU Energy Measurement via ZeusMonitor ---
        monitor = None
        if _ZEUS_AVAILABLE and torch.cuda.is_available():
            try:
                monitor = _ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
                monitor.begin_window("equest_vqc_training")
            except Exception:
                monitor = None

        sim_time_s = self.train()

        if monitor is not None:
            try:
                measurement = monitor.end_window("equest_vqc_training")
                real_energy_j = float(measurement.gpu_energy[0])
                zeus_window_s = float(measurement.time)
                energy_source = "zeus_gpu"
            except Exception:
                real_energy_j = sim_time_s * 15.0
                zeus_window_s = sim_time_s
                energy_source = "estimated"
        else:
            real_energy_j = sim_time_s * 15.0
            zeus_window_s = sim_time_s
            energy_source = "estimated"
        # --- End Feature 1 ---

        peak_memory_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        performance_metrics = self.evaluate()
        gate_counts = self.get_gate_counts()
        circuit_specs = self.get_circuit_specs()

        results = {
            "sim_time_gpu_s": sim_time_s,
            "real_energy_j": real_energy_j,
            "zeus_window_s": zeus_window_s,
            "energy_source": energy_source,
            "peak_memory_mb": peak_memory_mb,
            "total_calls": len(self.X_train_tensor) * self.epochs,
        }
        results.update(performance_metrics)
        results.update(gate_counts)
        results.update(circuit_specs)
        return results

