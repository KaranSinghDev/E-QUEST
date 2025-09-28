# E-QUEST Framework: Software Architecture

This document provides a **developer's-eye view** of the E-QUEST framework's architecture.  
It details the project’s structure, the role of each key component, and the design principles that ensure its modularity and robustness.

---

## 1. Guiding Principles

The framework was designed with the following principles in mind:

- **Modularity:**  
  Core logic (algorithms, data loaders) is decoupled from execution scripts.  
  → New algorithms can be added without modifying existing ones.

- **Reproducibility:**  
  A centralized configuration system ensures benchmarks can be run with precisely defined, version-controlled parameters.

- **Portability:**  
  Independent of the user’s operating system, with a robust environment definition and hardware-aware logic.

- **Usability:**  
  A top-level *conductor script* provides a simple, single-command entry point for running the full workflow.

---

## 2. Repository Structure

The project follows a standard **scientific software layout**:

```
E-QUEST_Final/
│
├── data/              # Raw input data (not version controlled).
├── docs/              # Documentation files (like this one).
├── results/           # Generated outputs (plots, reports).
├── scripts/           # User-facing Python scripts for analysis.
├── src/               # Core Python package (algorithms, loaders, configs).
├── tests/             # Standalone test scripts for components.
│
├── .gitignore         # Files ignored by Git.
├── environment.yml    # Conda environment definition.
├── README.md          # Project overview and installation guide.
└── run_full_analysis.py # Master "conductor" script.
```

---

## 3. The Core Library (`src/`)

The `src/` directory is the **heart of the framework**, structured as a proper Python package.

- **`base_algorithm.py`**  
  Defines the `BaseAlgorithm` abstract base class → blueprint for all algorithms.  
  Every algorithm must implement a `.benchmark()` method, making the system plug-and-play.

- **`classical_mlp.py`** & **`quantum_vqc.py`**  
  Concrete implementations of `BaseAlgorithm`.  
  Each file manages its **own data preparation, training, and metric calculation**.  
  Includes a **hardware-aware timer** in the `train()` method to select GPU or CPU timing automatically.

- **`data_loader.py`**  
  Handles all interactions with the raw **TrackML dataset**, using the official `trackml` library.

- **`config_loader.py`**  
  Provides `load_config()` function to locate and load the correct config file.  
  - Prefers the temporary `config.py` created by the conductor script.  
  - Falls back to `config_full.py`.  
  → Ensures every script can be run standalone for debugging.

- **`config_full.py`** & **`config_smoke_test.py`**  
  Master configuration templates:  
  - `config_full.py`: full scientific runs.  
  - `config_smoke_test.py`: lightweight diagnostic tests.

---

## 4. The Conductor Script (`run_full_analysis.py`)

This script is the **primary user interface** of the framework. Its design emphasizes robustness and user-friendliness.

- **Dual-Mode Execution**  
  Detects the `--smoke-test` flag:  
  - `True` → uses `config_smoke_test.py`.  
  - `False` → uses `config_full.py`.

- **Dynamic Configuration**  
  Creates a temporary `src/config.py` by copying the selected config.  
  Ensures all scripts can simply `from src import config`.

- **Sequential Workflow**  
  Runs the four main scripts from `scripts/` in predefined order via `subprocess`.  
  - Uses `check=True` → halts if any step fails.

- **Automated Reporting**  
  After benchmarks, triggers the reporting script to generate plots and a summary `final_report.md`.

- **Guaranteed Cleanup**  
  Uses `try...finally` to always delete the temporary `src/config.py`, leaving the repo clean.

---
