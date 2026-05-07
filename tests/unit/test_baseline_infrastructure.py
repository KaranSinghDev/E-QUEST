"""
Baseline tests: project infrastructure, config loader, and BaseAlgorithm contract.
These must pass at all times — they verify the framework's skeleton.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestBaseAlgorithmContract:
    """Verify the ABC enforces its contract correctly."""

    def test_cannot_instantiate_base_directly(self):
        from src.base_algorithm import BaseAlgorithm
        with pytest.raises(TypeError):
            BaseAlgorithm(config={})

    def test_subclass_without_benchmark_raises(self):
        from src.base_algorithm import BaseAlgorithm
        class IncompleteAlgo(BaseAlgorithm):
            def __init__(self, config):
                super().__init__(config)
            # missing benchmark()
        with pytest.raises(TypeError):
            IncompleteAlgo(config={})

    def test_valid_subclass_instantiates(self):
        from src.base_algorithm import BaseAlgorithm
        class MinimalAlgo(BaseAlgorithm):
            def __init__(self, config):
                super().__init__(config)
            def benchmark(self):
                return {}
        algo = MinimalAlgo(config={"key": "value"})
        assert algo.config == {"key": "value"}


class TestConfigLoader:
    """Verify config_loader finds and loads the right config file."""

    def test_loads_config_full_as_fallback(self):
        from src.config_loader import load_config
        config = load_config()
        assert config is not None

    def test_config_has_required_attributes(self):
        from src.config_loader import load_config
        config = load_config()
        required = [
            "DATA_DIR", "RAW_EVENTS_DIR", "RESULTS_DIR",
            "CLASSICAL_INPUT_SIZES", "QUANTUM_INPUT_SIZES",
            "MLP_EPOCHS", "VQC_EPOCHS",
            "ENERGY_PER_MAC_JOULES", "ENERGY_PER_1Q_GATE_J", "ENERGY_PER_2Q_GATE_J",
            "COMPUTATION_POWER_WATTS",
        ]
        for attr in required:
            assert hasattr(config, attr), f"Config missing required attribute: {attr}"

    def test_classical_input_sizes_nonempty_list(self):
        from src.config_loader import load_config
        config = load_config()
        assert isinstance(config.CLASSICAL_INPUT_SIZES, list)
        assert len(config.CLASSICAL_INPUT_SIZES) > 0

    def test_quantum_input_sizes_nonempty_list(self):
        from src.config_loader import load_config
        config = load_config()
        assert isinstance(config.QUANTUM_INPUT_SIZES, list)
        assert len(config.QUANTUM_INPUT_SIZES) > 0

    def test_energy_constants_are_positive(self):
        from src.config_loader import load_config
        config = load_config()
        assert config.ENERGY_PER_MAC_JOULES > 0
        assert config.ENERGY_PER_1Q_GATE_J > 0
        assert config.ENERGY_PER_2Q_GATE_J > 0
        assert config.COMPUTATION_POWER_WATTS > 0

    def test_2q_gate_more_expensive_than_1q(self):
        """Physical constraint: 2-qubit gates always cost more energy than 1-qubit."""
        from src.config_loader import load_config
        config = load_config()
        assert config.ENERGY_PER_2Q_GATE_J > config.ENERGY_PER_1Q_GATE_J


class TestSyntheticDataFixtures:
    """Verify the test fixtures themselves produce valid data."""

    def test_synthetic_df_has_correct_columns(self, synthetic_df_small):
        assert set(synthetic_df_small.columns) == {"delta_r", "delta_phi", "delta_z", "label"}

    def test_synthetic_df_is_balanced(self, synthetic_df_small):
        counts = synthetic_df_small["label"].value_counts()
        assert counts[0] == counts[1], "Fixture must be perfectly balanced"

    def test_synthetic_df_labels_are_binary(self, synthetic_df_small):
        assert set(synthetic_df_small["label"].unique()).issubset({0, 1})

    def test_synthetic_csv_exists_and_readable(self, synthetic_csv_path):
        import pandas as pd
        assert os.path.exists(synthetic_csv_path)
        df = pd.read_csv(synthetic_csv_path)
        assert len(df) > 0
        assert "label" in df.columns

    def test_phi_cut_range_realistic(self, synthetic_df_small):
        """delta_phi values should be within the cone search cut of ±0.1 rad."""
        assert synthetic_df_small["delta_phi"].abs().max() < 0.1
