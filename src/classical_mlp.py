import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score # CHANGED: Add new imports
import time

# --- Universal Path Setup ---
import sys
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if project_root not in sys.path:
        sys.path.append(project_root)

    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

except NameError:
    if '.' not in sys.path:
        sys.path.append('.')
# --- End of Universal Path Setup ---

# ✅ Import runtime mode safely
try:
    from src.config import DATA_PREPROCESSING_MODE
except Exception as e:
    print("❌ Could not import DATA_PREPROCESSING_MODE from src.config")
    print("   Did run_benchmark.py create src/config.py?")
    print(f"   Error: {e}")
    sys.exit(1)


# Import our main blueprint
from src.base_algorithm import BaseAlgorithm

# --- Part 1: Define the Neural Network using PyTorch ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(SimpleMLP, self).__init__()

        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.output_layer(x)

        # Important:
        # undersample mode expects probabilities
        # weighted mode expects raw logits
        if DATA_PREPROCESSING_MODE == "undersample":
            return torch.sigmoid(x)
        else:
            return x

# --- Part 2: The Main Algorithm Wrapper Class ---
class ClassicalMLP(BaseAlgorithm):
    """
    A wrapper for our simple MLP model. This class handles data loading,
    training, prediction, and benchmarking, conforming to the BaseAlgorithm interface.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  -> Using device: {self.device}")
        self.model = SimpleMLP().to(self.device)
        self.dataset_path = self.config.get("dataset_path")
        self.epochs = self.config.get("epochs", 3)
        self.learning_rate = self.config.get("lr", 0.001)
        self.batch_size = self.config.get("batch_size", 4096)

    def _load_and_prepare_data(self):
        """Loads the dataset and splits it into training and validation sets."""
        print("  -> Loading and preparing ML dataset...")
        df = pd.read_csv(self.dataset_path)
        X = df[['delta_r', 'delta_phi', 'delta_z']].values
        y = df['label'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        pos = (y == 1).sum()
        neg = (y == 0).sum()

        if DATA_PREPROCESSING_MODE == "weighted":
            if pos == 0:
                pos_weight_value = 1.0
            else:
                pos_weight_value = neg / pos

            self.pos_weight_tensor = torch.tensor(
                [pos_weight_value], dtype=torch.float32
            ).to(self.device)

            print(f"  -> Class distribution: pos={pos}, neg={neg}, pos_weight={pos_weight_value:.2f}")
        else:
            print(f"  -> Class distribution: pos={pos}, neg={neg} (undersample mode)")

        print(f"  -> Data ready. Training samples: {len(self.X_train_tensor)}, Validation samples: {len(self.X_val_tensor)}")

    def train(self) -> float:
            """
            Trains the PyTorch model and returns the total GPU computation time in seconds.
            """
            train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            if DATA_PREPROCESSING_MODE == "weighted":
                criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
            else:
                criterion = nn.BCELoss()

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # --- PRECISE GPU TIME MEASUREMENT ---
            # Create CUDA events for accurate timing, ignoring CPU data loading time.
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            total_gpu_time_ms = 0
            # --- END ---

            print(f"  -> Starting training for {self.epochs} epochs...")
            self.model.train()
            for epoch in range(self.epochs):
                epoch_wall_time_start = time.time() # For user feedback only
                
                for features, labels in train_loader:
                    features = features.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # --- Record GPU time for this batch ---
                    start_event.record()
                    
                    # Computation
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    end_event.record()
                    torch.cuda.synchronize() # Wait for the GPU to finish this batch
                    total_gpu_time_ms += start_event.elapsed_time(end_event)
                    # --- End of recording ---

                print(f"     Epoch [{epoch+1}/{self.epochs}], Wall Clock Time: {time.time() - epoch_wall_time_start:.2f}s")
            
            # Return total GPU time in seconds
            return total_gpu_time_ms / 1000.0

    def evaluate(self) -> dict:
        """
        Evaluates the model on the validation set and returns a dictionary of
        performance metrics: AUC, Precision, and Recall.
        """
        self.model.eval()
        X_val_device = self.X_val_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(X_val_device).cpu()

        # Convert to probabilities
        if DATA_PREPROCESSING_MODE == "weighted":
            logits = outputs.numpy()
            predictions_proba = 1 / (1 + np.exp(-logits))   # sigmoid manually

        else:
            predictions_proba = outputs.numpy()

        
        # Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
        predictions_binary = (predictions_proba > 0.5).astype(int)
        
        # Get the ground truth labels
        true_labels = self.y_val_tensor.numpy()
        
        # Calculate all metrics
        auc_score = roc_auc_score(true_labels, predictions_proba)
        precision = precision_score(true_labels, predictions_binary, zero_division=0)
        recall = recall_score(true_labels, predictions_binary, zero_division=0)


        # Return all metrics in a dictionary
        return {
            "accuracy_auc": auc_score,
            "precision": precision,
            "recall": recall
        }

    def benchmark(self) -> dict:
        """
        Orchestrates the benchmark, measuring time, memory, and a full suite of
        performance metrics (AUC, Precision, Recall).
        """
        self._load_and_prepare_data()
        gpu_train_time_s = self.train()
        
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        
        # The evaluate() method now returns a dictionary of all performance scores
        performance_metrics = self.evaluate()
        
        # Combine all hardware and performance metrics into a single results dictionary
        results = {
            "time_training_gpu_s": gpu_train_time_s,
            "peak_memory_mb": peak_memory_mb
        }
        results.update(performance_metrics) # Cleanly merge the two dictionaries
        
        return results
