"""
Shared pytest fixtures for E-QUEST unit tests.

All fixtures use synthetic data so tests run without needing the real
TrackML dataset on disk. The synthetic data has the same schema as
the real preprocessed dataset: columns [delta_r, delta_phi, delta_z, label].
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is on sys.path so `src` imports work when pytest
# is run from any directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

N_SAMPLES_SMALL = 200       # balanced, used by VQC (fast)
N_SAMPLES_CLASSICAL = 2000  # balanced, used by MLP (still fast)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_segment_df(n_samples: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Creates a synthetic but realistic balanced segment DataFrame.
    Half the rows are 'true' segments (label=1), half are 'false' (label=0).
    Feature ranges match typical TrackML cone-search output.
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2

    # True segments: small deltas (geometrically consistent)
    true_r   = rng.normal(loc=20.0,  scale=5.0,   size=half)
    true_phi = rng.normal(loc=0.0,   scale=0.02,  size=half)
    true_z   = rng.normal(loc=0.0,   scale=20.0,  size=half)

    # False segments: larger, noisier deltas
    false_r   = rng.uniform(low=-50.0, high=50.0, size=half)
    false_phi = rng.uniform(low=-0.09, high=0.09, size=half)
    false_z   = rng.uniform(low=-190.0, high=190.0, size=half)

    df = pd.DataFrame({
        "delta_r":   np.concatenate([true_r,   false_r]),
        "delta_phi": np.concatenate([true_phi, false_phi]),
        "delta_z":   np.concatenate([true_z,   false_z]),
        "label":     np.concatenate([np.ones(half), np.zeros(half)]).astype(int),
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_df_small():
    """Balanced synthetic segment DataFrame — small, for VQC tests."""
    return _make_synthetic_segment_df(N_SAMPLES_SMALL)


@pytest.fixture(scope="session")
def synthetic_df_classical():
    """Balanced synthetic segment DataFrame — larger, for MLP tests."""
    return _make_synthetic_segment_df(N_SAMPLES_CLASSICAL)


@pytest.fixture(scope="session")
def synthetic_csv_path(tmp_path_factory, synthetic_df_classical):
    """
    Writes the classical synthetic DataFrame to a temporary CSV file and
    returns the path. Used by ClassicalMLP which reads from disk.
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    csv_path = str(tmp_dir / "synthetic_segments.csv")
    synthetic_df_classical.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="session")
def mlp_config(synthetic_csv_path):
    """Minimal ClassicalMLP config: 1 epoch, small batch, synthetic data."""
    return {
        "dataset_path": synthetic_csv_path,
        "epochs": 1,
        "lr": 0.001,
        "batch_size": 512,
    }


@pytest.fixture(scope="session")
def vqc_config(synthetic_df_small):
    """Minimal QuantumVQC config: 1 epoch, tiny batch, 20 samples, synthetic data."""
    return {
        "dataset": synthetic_df_small,
        "num_samples": 20,
        "epochs": 1,
        "lr": 0.01,
        "batch_size": 5,
        "num_layers": 1,
    }


@pytest.fixture(scope="session")
def mlp_instance(mlp_config):
    """An initialized ClassicalMLP ready for benchmarking."""
    from src.classical_mlp import ClassicalMLP
    return ClassicalMLP(config=mlp_config)


@pytest.fixture(scope="session")
def vqc_instance(vqc_config):
    """An initialized QuantumVQC ready for benchmarking."""
    from src.quantum_vqc import QuantumVQC
    torch.manual_seed(RANDOM_SEED)
    return QuantumVQC(config=vqc_config)
