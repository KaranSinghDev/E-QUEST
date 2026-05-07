"""
Baseline tests for the QuantumVQC algorithm.
Uses 20-sample synthetic data and 1 epoch to stay fast.
"""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.mark.quantum
@pytest.mark.slow
class TestQuantumVQCInterface:
    """Test that QuantumVQC fulfills the BaseAlgorithm contract."""

    def test_inherits_base_algorithm(self):
        from src.base_algorithm import BaseAlgorithm
        from src.quantum_vqc import QuantumVQC
        assert issubclass(QuantumVQC, BaseAlgorithm)

    def test_instantiation_with_minimal_config(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        algo = QuantumVQC(config=vqc_config)
        assert algo is not None
        assert algo.epochs == vqc_config["epochs"]

    def test_weights_are_trainable_tensor(self, vqc_instance):
        assert isinstance(vqc_instance.weights, torch.Tensor)
        assert vqc_instance.weights.requires_grad

    def test_weights_shape_matches_layers_and_wires(self, vqc_config):
        import pennylane as qml
        from src.quantum_vqc import QuantumVQC, NUM_QUBITS
        algo = QuantumVQC(config=vqc_config)
        expected_shape = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=vqc_config["num_layers"], n_wires=NUM_QUBITS
        )
        assert algo.weights.shape == expected_shape

    def test_benchmark_returns_dict(self, vqc_instance):
        result = vqc_instance.benchmark()
        assert isinstance(result, dict)

    def test_benchmark_has_required_keys(self, vqc_instance):
        result = vqc_instance.benchmark()
        required_keys = {
            "sim_time_gpu_s",
            "peak_memory_mb",
            "accuracy_auc",
            "precision",
            "recall",
            "n_1q_gates",
            "n_2q_gates",
            "circuit_depth",
            "total_calls",
        }
        missing = required_keys - result.keys()
        assert not missing, f"benchmark() result missing keys: {missing}"

    def test_sim_time_is_positive(self, vqc_instance):
        result = vqc_instance.benchmark()
        assert result["sim_time_gpu_s"] > 0

    def test_auc_is_valid_probability(self, vqc_instance):
        result = vqc_instance.benchmark()
        assert 0.0 <= result["accuracy_auc"] <= 1.0

    def test_precision_recall_valid_range(self, vqc_instance):
        result = vqc_instance.benchmark()
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_total_calls_equals_samples_times_epochs(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        torch.manual_seed(42)
        algo = QuantumVQC(config=vqc_config)
        result = algo.benchmark()
        # total_calls = training_samples * epochs (80% of num_samples = train set)
        expected_train_samples = int(vqc_config["num_samples"] * 0.8)
        expected_calls = expected_train_samples * vqc_config["epochs"]
        assert result["total_calls"] == expected_calls, (
            f"Expected total_calls={expected_calls}, got {result['total_calls']}"
        )


@pytest.mark.quantum
class TestQuantumVQCCircuitAnalysis:
    """Test gate counting and circuit depth — the hardware-independent metrics."""

    def test_gate_counts_nonzero(self, vqc_instance):
        """After data prep, gate counts must be positive integers."""
        vqc_instance._load_and_prepare_data()
        counts = vqc_instance.get_gate_counts()
        assert counts["n_1q_gates"] > 0, "Must have at least 1 single-qubit gate"
        assert counts["n_2q_gates"] > 0, "Must have at least 1 two-qubit gate"

    def test_gate_counts_return_ints(self, vqc_instance):
        vqc_instance._load_and_prepare_data()
        counts = vqc_instance.get_gate_counts()
        assert isinstance(counts["n_1q_gates"], int)
        assert isinstance(counts["n_2q_gates"], int)

    def test_circuit_depth_positive(self, vqc_instance):
        vqc_instance._load_and_prepare_data()
        specs = vqc_instance.get_circuit_specs()
        assert specs["circuit_depth"] > 0

    def test_more_1q_gates_than_2q_gates(self, vqc_instance):
        """StronglyEntanglingLayers uses more single-qubit than two-qubit gates."""
        vqc_instance._load_and_prepare_data()
        counts = vqc_instance.get_gate_counts()
        assert counts["n_1q_gates"] > counts["n_2q_gates"], (
            "StronglyEntanglingLayers ansatz should have more 1q than 2q gates"
        )

    def test_gate_counts_deterministic(self, vqc_config):
        """Gate counts must be identical across two runs (circuit structure is fixed)."""
        from src.quantum_vqc import QuantumVQC
        algo1 = QuantumVQC(config=vqc_config)
        algo1._load_and_prepare_data()

        algo2 = QuantumVQC(config=vqc_config)
        algo2._load_and_prepare_data()

        counts1 = algo1.get_gate_counts()
        counts2 = algo2.get_gate_counts()
        assert counts1 == counts2, (
            "Gate counts must be deterministic — they depend on circuit structure, not weights"
        )

    def test_circuit_depth_deterministic(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        algo1 = QuantumVQC(config=vqc_config)
        algo1._load_and_prepare_data()
        algo2 = QuantumVQC(config=vqc_config)
        algo2._load_and_prepare_data()
        assert algo1.get_circuit_specs() == algo2.get_circuit_specs()
