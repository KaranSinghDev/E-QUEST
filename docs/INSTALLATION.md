# E-QUEST Framework: Installation Guide

This guide provides detailed, step-by-step instructions for setting up the **E-QUEST framework** and its Conda environment on both **Windows** and **Linux (including WSL)** systems.

---

## 1. Prerequisites

Before you begin, please ensure you have the following software installed on your system:

- **Git**: For cloning the project repository.  
  👉 [Download Git](https://git-scm.com/)  

- **Conda**: For managing the Python environment. We recommend installing **[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)** for a lightweight installation.

- **For GPU Acceleration (Optional but Recommended):**
  - An **NVIDIA GPU** with a modern architecture (e.g., Turing, Ampere, Ada Lovelace).
  - The latest **NVIDIA Graphics Driver** installed (required for CUDA).

---

## 2. Standard Installation (CPU Support)

These steps will create a complete and functional environment that provides high-performance **CPU-based quantum simulation**.  
The framework is guaranteed to work in this default configuration.

### Step 2.1: Clone the Repository

Open your terminal (**Anaconda Prompt** on Windows, or a standard terminal on Linux/WSL) and run:

```bash
git clone https://github.com/your-username/E-QUEST_Final.git
cd E-QUEST_Final
```

---

### Step 2.2: Create the Conda Environment

Use the provided `environment.yml` file to create the isolated Python environment.  
This command will download and install all required libraries (may take several minutes):

```bash
conda env create -f environment.yml
```

---

### Step 2.3: Activate the Environment

You must activate the environment every time you want to use the framework:

```bash
conda activate equest-env
```

✅ **Verification**: Your terminal prompt should now be prefixed with `(equest-env)`.  

---

## 3. [Optional] Enabling GPU Acceleration

For maximum performance, users with a compatible **NVIDIA GPU** can enable the GPU-accelerated backend for quantum simulations.  
⚠️ **Note**: Complete the standard installation (Section 2) and activate the `equest-env` before proceeding.

---

### 3.1 Windows Users

The GPU backend on Windows requires manual installation of the **NVIDIA cuQuantum SDK**.

1. **Download and Install NVIDIA cuQuantum SDK**:
   - Navigate to the [NVIDIA cuQuantum SDK homepage](https://developer.nvidia.com/cuquantum-sdk).
   - Download the latest installer appropriate for your system (**Windows, CUDA 12.x**).
   - Run the installer and follow its instructions.

2. **Install PennyLane GPU Bindings**:  
   Inside your activated environment, run:

   ```bash
   pip install "pennylane-lightning[gpu]==0.42.0"
   ```

---

### 3.2 Linux / WSL Users

On Linux/WSL, the dependencies are available directly via pip.

Install the PennyLane GPU package by running:

```bash
pip install "pennylane-lightning[gpu]==0.42.0"
```

---

## 4. Verifying the Installation

After completing installation, run a **smoke test** to confirm that everything works.

From the project’s root directory:

```bash
python run_benchmark.py --smoke-test
```

### ✅ Expected Output:

- **If GPU support is enabled**:
  ```
  SUCCESS: PennyLane is using 'lightning.gpu' with 4 qubits.
  ```

- **If CPU-only installation is used**:
  ```
  WARNING: 'lightning.gpu' backend not found.
  SUCCESS: Falling back to high-performance CPU backend 'lightning.qubit'.
  ```

If the script ends with:

```
🎉 ENTIRE WORKFLOW COMPLETED SUCCESSFULLY! 🎉
```

your installation is correct, and the framework is ready to use. 🚀

---
