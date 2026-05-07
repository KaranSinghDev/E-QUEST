"""
Baseline tests for the ClassicalMLP algorithm.
Uses synthetic data so no TrackML download is needed.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.mark.classical
@pytest.mark.slow
class TestClassicalMLPInterface:
    """Test that ClassicalMLP fulfills the BaseAlgorithm contract."""

    def test_inherits_base_algorithm(self):
        from src.base_algorithm import BaseAlgorithm
        from src.classical_mlp import ClassicalMLP
        assert issubclass(ClassicalMLP, BaseAlgorithm)

    def test_instantiation_with_minimal_config(self, mlp_config):
        from src.classical_mlp import ClassicalMLP
        algo = ClassicalMLP(config=mlp_config)
        assert algo is not None
        assert algo.epochs == mlp_config["epochs"]
        assert algo.learning_rate == mlp_config["lr"]
        assert algo.batch_size == mlp_config["batch_size"]

    def test_benchmark_returns_dict(self, mlp_instance):
        result = mlp_instance.benchmark()
        assert isinstance(result, dict), "benchmark() must return a dict"

    def test_benchmark_has_required_keys(self, mlp_instance):
        result = mlp_instance.benchmark()
        required_keys = {
            "time_training_gpu_s",
            "peak_memory_mb",
            "accuracy_auc",
            "precision",
            "recall",
        }
        missing = required_keys - result.keys()
        assert not missing, f"benchmark() result missing keys: {missing}"

    def test_training_time_is_positive(self, mlp_instance):
        result = mlp_instance.benchmark()
        assert result["time_training_gpu_s"] > 0, "Training time must be > 0"

    def test_memory_is_nonnegative(self, mlp_instance):
        result = mlp_instance.benchmark()
        assert result["peak_memory_mb"] >= 0, "Memory must be >= 0"

    def test_auc_is_valid_probability(self, mlp_instance):
        result = mlp_instance.benchmark()
        auc = result["accuracy_auc"]
        assert 0.0 <= auc <= 1.0, f"AUC must be in [0, 1], got {auc}"

    def test_auc_beats_random_on_separable_data(self, mlp_instance):
        """MLP on structured synthetic data should do better than random guessing."""
        result = mlp_instance.benchmark()
        assert result["accuracy_auc"] > 0.5, (
            f"MLP AUC {result['accuracy_auc']:.3f} should beat random (0.5) "
            "on synthetic data with clear signal"
        )

    def test_precision_is_valid_probability(self, mlp_instance):
        result = mlp_instance.benchmark()
        assert 0.0 <= result["precision"] <= 1.0

    def test_recall_is_valid_probability(self, mlp_instance):
        result = mlp_instance.benchmark()
        assert 0.0 <= result["recall"] <= 1.0


@pytest.mark.classical
class TestClassicalMLPDataPreparation:
    """Test the internal data loading/preparation step."""

    def test_load_and_prepare_creates_tensors(self, mlp_config):
        import torch
        from src.classical_mlp import ClassicalMLP
        algo = ClassicalMLP(config=mlp_config)
        algo._load_and_prepare_data()
        assert hasattr(algo, "X_train_tensor")
        assert hasattr(algo, "X_val_tensor")
        assert isinstance(algo.X_train_tensor, torch.Tensor)
        assert isinstance(algo.X_val_tensor, torch.Tensor)

    def test_feature_tensor_has_3_columns(self, mlp_config):
        from src.classical_mlp import ClassicalMLP
        algo = ClassicalMLP(config=mlp_config)
        algo._load_and_prepare_data()
        assert algo.X_train_tensor.shape[1] == 3, "Must have exactly 3 features"

    def test_train_val_split_roughly_80_20(self, mlp_config):
        from src.classical_mlp import ClassicalMLP
        algo = ClassicalMLP(config=mlp_config)
        algo._load_and_prepare_data()
        total = len(algo.X_train_tensor) + len(algo.X_val_tensor)
        val_fraction = len(algo.X_val_tensor) / total
        assert 0.18 <= val_fraction <= 0.22, (
            f"Val split should be ~20%, got {val_fraction:.2%}"
        )
