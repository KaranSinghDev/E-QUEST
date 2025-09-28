import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import time
# (at the top with the other imports)
from pennylane import numpy as np

# Import PennyLane and our main blueprint
import pennylane as qml
from src.base_algorithm import BaseAlgorithm

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
        self.dataset_path = self.config.get("dataset_path")
        self.epochs = self.config.get("epochs", 3)
        self.learning_rate = self.config.get("lr", 0.01)
        self.batch_size = self.config.get("batch_size", 128)
        
        # --- Define the shape of our trainable weights ---
        num_layers = self.config.get("num_layers", 2)
        self.weights_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=NUM_QUBITS)
        
        # --- THIS IS THE FIX ---
        # 1. First, determine the target device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  -> PyTorch components (optimizer) will use device: {self.device}")
        
        # 2. Now, create the weights tensor DIRECTLY on the target device.
        #    This ensures it is a "leaf" tensor on the correct device from the start.
        self.weights = torch.randn(self.weights_shape, device=self.device, dtype=torch.float64, requires_grad=True)
        # --- END OF FIX ---

    def _load_and_prepare_data(self):
        """
        Loads and prepares a smaller subset of data suitable for the VQC.
        This version now loads the full dataset to ensure it can create a
        balanced subset, preventing sampling errors.
        """
        print("  -> Loading and preparing ML dataset from the main data file...")
        
        # --- THIS IS THE DEFINITIVE FIX ---
        # 1. ALWAYS load the main, full dataset to ensure we have a large pool of samples.
        #    We will ignore the self.dataset_path from the config for the purpose of loading.
        main_dataset_path = "track_segments_ml_dataset.csv"
        try:
            df = pd.read_csv(main_dataset_path)
        except FileNotFoundError:
            print(f"❌ CRITICAL ERROR: The main dataset '{main_dataset_path}' was not found.")
            print("   Please run 'create_ml_dataset.py' first.")
            # Create empty tensors to prevent further crashes down the line
            self.X_train_tensor = torch.empty(0)
            self.y_train_tensor = torch.empty(0)
            self.X_val_tensor = torch.empty(0)
            self.y_val_tensor = torch.empty(0)
            return

        # 2. Get the requested number of samples for THIS SPECIFIC benchmark run from the config.
        num_samples = self.config.get("num_samples", 2000)
        print(f"  -> VQC is slow. Creating a balanced subset of {num_samples} total samples.")
        
        # 3. Securely sample from the large DataFrame. This is now guaranteed to work.
        num_true_needed = num_samples // 2
        num_false_needed = num_samples - num_true_needed # Handles odd numbers correctly
        
        # Check if we have enough samples in the full dataset
        if len(df[df['label'] == 1]) < num_true_needed or len(df[df['label'] == 0]) < num_false_needed:
            print(f"❌ CRITICAL ERROR: Full dataset does not contain enough samples to create a balanced set of size {num_samples}.")
            # Handle error appropriately
            return

        true_segments = df[df['label'] == 1].sample(n=num_true_needed, random_state=42)
        false_segments = df[df['label'] == 0].sample(n=num_false_needed, random_state=42)
        # --- END OF FIX ---
        
        balanced_df = pd.concat([true_segments, false_segments]).sample(frac=1, random_state=42) # Shuffle
        
        X = balanced_df[['delta_r', 'delta_phi', 'delta_z']].values
        y = balanced_df['label'].values
        
        y = y * 2 - 1 # Rescale y from {0, 1} to {-1, 1}
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float64)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float64).view(-1, 1)
        
        print(f"  -> Data ready. Training samples: {len(self.X_train_tensor)}, Validation samples: {len(self.X_val_tensor)}")

    # REPLACE your existing train() method with this one
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

    # REPLACE the existing evaluate method with this one
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

        # --- ADDED: Calculate All Metrics ---
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
        # --- END of ADDED Block ---

        # Return all metrics in a dictionary
        return {
            "accuracy_auc": auc_score,
            "precision": precision,
            "recall": recall
        }

 
        # ... (no changes here) ...

    # REPLACE your existing get_gate_counts method with this one
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
        
        return {"1q_gates_per_call": total_1q_gates, "2q_gates_per_call": total_2q_gates}

    # ADD the following NEW method right after get_gate_counts
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

    # REPLACE your existing benchmark method with this one
    def benchmark(self) -> dict:
        """
        Orchestrates the full benchmark for the VQC, now measuring time,
        accuracy, peak memory, gate counts, and circuit depth.
        """
        self._load_and_prepare_data()
        gpu_train_time_s = self.train()

        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        
        # The evaluate() method now returns a dictionary of all performance scores
        performance_metrics = self.evaluate()
        gate_counts = self.get_gate_counts()
        circuit_specs = self.get_circuit_specs()
        
        # Combine all hardware and performance metrics into a single results dictionary
        results = {
            "time_training_gpu_s": gpu_train_time_s,
            "peak_memory_mb": peak_memory_mb,
            "total_training_calls": len(self.X_train_tensor) * self.epochs
        }
        # Cleanly merge the dictionaries from our helper methods
        results.update(performance_metrics)
        results.update(gate_counts)
        results.update(circuit_specs)
        
        return results

