# Extending the E-QUEST Framework

One of the core design goals of the E-QUEST framework is modularity. The system is designed to be easily extended with new classical or quantum algorithms. This document provides a conceptual guide and a high-level tutorial on how to add a new algorithm to the benchmark.

---

## The `BaseAlgorithm` Interface

The key to the framework's modularity is the `BaseAlgorithm` interface, defined in `src/base_algorithm.py`. This is an "abstract base class" that acts as a contract. To add a new algorithm, you simply need to create a new class that promises to fulfill this contract.

The contract has two requirements:

1. **An `__init__(self, config: dict)` method:** The constructor for your class, which will receive a configuration dictionary.
2. **A `benchmark(self) -> dict` method:** The main execution method. This method must perform the entire benchmark for your algorithm and **return a dictionary** containing the measured results.

For the reporting scripts to work correctly, the keys in the returned dictionary should be consistent (e.g., `accuracy_auc`, `peak_memory_mb`, etc.).

---

## Tutorial: How to Add a New Algorithm

Here is the general pattern to follow. As an example, imagine we want to add a `LogisticRegression` model.

### Step 1: Create the Algorithm Wrapper Class

The first step is to create a new Python file in the `src/` directory (e.g., `src/classical_lr.py`). Inside this file, you will define your new class.

The structure of this class is very simple. You just need to inherit from `BaseAlgorithm` and implement the two required methods.

**Conceptual Example:**

```python
# In: src/classical_lr.py

# Import the base class and any libraries you need (e.g., sklearn)
from src.base_algorithm import BaseAlgorithm
from sklearn.linear_model import LogisticRegression
# ... other imports for metrics, pandas, etc.

class ClassicalLR(BaseAlgorithm):
    """
    A wrapper for a scikit-learn Logistic Regression model.
    """
    def __init__(self, config: dict):
        # Always call the parent constructor
        super().__init__(config)
        # Your specific initialization logic here
        self.model = LogisticRegression()
        # ...

    def benchmark(self) -> dict:
        # 1. Load and prepare data using the path from the config.
        # ... your data loading logic ...

        # 2. Perform the actual benchmark (e.g., train the model).
        # ... your timing and training logic ...

        # 3. Evaluate the model's performance.
        # ... your logic for calculating AUC, precision, etc. ...

        # 4. Return the results in a dictionary with standard keys.
        return {
            "computation_time_s": ...,
            "accuracy_auc": ...,
            "precision": ...,
            "recall": ...,
            "peak_memory_mb": ...
        }
```

### Step 2: Create the Analysis Script

Next, you need a script in the `scripts/` directory to drive the benchmark for your new algorithm.

The easiest and best way to do this is to copy an existing analysis script and adapt it. For example, you could:

1. Copy `scripts/analyze_scaling_classical.py` to `scripts/analyze_scaling_lr.py`.
2. Open the new file.
3. Change the import line from `from src.classical_mlp import ClassicalMLP` to `from src.classical_lr import ClassicalLR`.
4. In the `run_empirical_benchmark` function, change the line that creates the algorithm object from `mlp_algo = ClassicalMLP(...)` to `lr_algo = ClassicalLR(...)`.

The rest of the script's logic (looping through input sizes, saving results to a CSV) can likely be reused directly.

### Step 3: Update the Reporting

Finally, to include your new algorithm in the final comparison plots, you can modify the `scripts/generate_report.py` script.

1. Load the new results CSV file (e.g., `lr_results.csv`) into a new pandas DataFrame.
2. In the plotting functions, add new `ax.plot(...)` calls to draw the performance and resource curves for your new algorithm. Be sure to give it a unique color and marker for clarity.

By following this three-step pattern, any new algorithm can be cleanly and robustly integrated into the E-QUEST framework's benchmark and analysis pipeline.