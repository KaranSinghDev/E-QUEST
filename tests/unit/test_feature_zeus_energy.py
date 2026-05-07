"""
Tests for Feature 1: Real GPU energy measurement via ZeusMonitor.

These tests verify that:
1. benchmark() now returns a 'real_energy_j' key with a positive value.
2. The real energy is physically plausible (> 0, not absurdly large).
3. The fallback (no GPU / Zeus unavailable) still returns a value.
4. Existing keys are not broken by the change.
5. The energy is consistent with the measured time × known GPU TDP range.
"""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RTX_3060_TDP_WATTS = 115.0  # Max TDP for RTX 3060 Laptop GPU
RTX_3060_MIN_WATTS = 5.0    # Minimum idle power


@pytest.mark.classical
@pytest.mark.slow
@pytest.mark.gpu
class TestZeusEnergyClassicalMLP:
    """Verify Zeus energy measurement in ClassicalMLP."""

    def test_benchmark_returns_real_energy_key(self, mlp_config):
        from src.classical_mlp import ClassicalMLP
        result = ClassicalMLP(config=mlp_config).benchmark()
        assert "real_energy_j" in result, (
            "benchmark() must return 'real_energy_j' after Feature 1 implementation"
        )

    def test_real_energy_is_positive(self, mlp_config):
        from src.classical_mlp import ClassicalMLP
        result = ClassicalMLP(config=mlp_config).benchmark()
        assert result["real_energy_j"] > 0.0, "Real energy must be positive"

    def test_real_energy_physically_plausible(self, mlp_config):
        """
        Implied average GPU power (energy / zeus_window_s) must be within
        physical GPU power bounds. RTX 3060 Laptop GPU: min ~5W, max ~115W.
        We allow 2× headroom above TDP for transient spikes.
        """
        from src.classical_mlp import ClassicalMLP
        result = ClassicalMLP(config=mlp_config).benchmark()
        energy_j = result["real_energy_j"]
        window_s = result["zeus_window_s"]
        assert window_s > 0, "zeus_window_s must be positive"
        avg_power_w = energy_j / window_s
        assert RTX_3060_MIN_WATTS <= avg_power_w <= RTX_3060_TDP_WATTS * 2.0, (
            f"Implied average power {avg_power_w:.1f} W is outside physical GPU range "
            f"[{RTX_3060_MIN_WATTS}W, {RTX_3060_TDP_WATTS * 2.0}W]"
        )

    def test_real_energy_not_suspiciously_zero(self, mlp_config):
        """Flat-constant fallback returns time×15W — real Zeus value should differ."""
        from src.classical_mlp import ClassicalMLP
        result = ClassicalMLP(config=mlp_config).benchmark()
        flat_estimate = result["time_training_gpu_s"] * 15.0
        # Zeus should give a different (and typically larger) reading than 15W flat estimate
        assert result["real_energy_j"] != flat_estimate, (
            "real_energy_j appears to be the old flat-constant estimate — "
            "Zeus hardware measurement should differ"
        )

    def test_existing_keys_still_present(self, mlp_config):
        """Adding Zeus must not remove any previously existing result keys."""
        from src.classical_mlp import ClassicalMLP
        result = ClassicalMLP(config=mlp_config).benchmark()
        original_keys = {
            "time_training_gpu_s", "peak_memory_mb",
            "accuracy_auc", "precision", "recall",
        }
        missing = original_keys - result.keys()
        assert not missing, f"Feature 1 broke existing keys: {missing}"

    def test_measured_energy_source_is_labeled(self, mlp_config):
        """benchmark() should report HOW energy was measured (real vs estimated)."""
        from src.classical_mlp import ClassicalMLP
        result = ClassicalMLP(config=mlp_config).benchmark()
        assert "energy_source" in result, (
            "Result must include 'energy_source' key: 'zeus_gpu' or 'estimated'"
        )
        assert result["energy_source"] in ("zeus_gpu", "estimated"), (
            f"Unexpected energy_source value: {result['energy_source']!r}"
        )

    def test_energy_source_is_zeus_when_cuda_available(self, mlp_config):
        from src.classical_mlp import ClassicalMLP
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
        result = ClassicalMLP(config=mlp_config).benchmark()
        assert result["energy_source"] == "zeus_gpu", (
            "When CUDA is available, energy_source must be 'zeus_gpu'"
        )


@pytest.mark.quantum
@pytest.mark.slow
@pytest.mark.gpu
class TestZeusEnergyQuantumVQC:
    """Verify Zeus energy measurement in QuantumVQC."""

    def test_benchmark_returns_real_energy_key(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        torch.manual_seed(42)
        result = QuantumVQC(config=vqc_config).benchmark()
        assert "real_energy_j" in result

    def test_real_energy_is_positive(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        torch.manual_seed(42)
        result = QuantumVQC(config=vqc_config).benchmark()
        assert result["real_energy_j"] > 0.0

    def test_energy_source_reported(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        torch.manual_seed(42)
        result = QuantumVQC(config=vqc_config).benchmark()
        assert "energy_source" in result
        assert result["energy_source"] in ("zeus_gpu", "estimated")

    def test_existing_quantum_keys_still_present(self, vqc_config):
        from src.quantum_vqc import QuantumVQC
        torch.manual_seed(42)
        result = QuantumVQC(config=vqc_config).benchmark()
        original_keys = {
            "sim_time_gpu_s", "peak_memory_mb", "accuracy_auc",
            "precision", "recall", "n_1q_gates", "n_2q_gates",
            "circuit_depth", "total_calls",
        }
        missing = original_keys - result.keys()
        assert not missing, f"Feature 1 broke existing quantum keys: {missing}"
