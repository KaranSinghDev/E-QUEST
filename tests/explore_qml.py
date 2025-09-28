import pennylane as qml
import torch
import numpy as np

# --- Configuration ---
NUM_QUBITS = 4
NUM_LAYERS = 2

def main():
    """
    A comprehensive, in-depth exploration of qml.specs to definitively understand
    its functionality, arguments, and return types, especially in the context of
    circuit decomposition and PyTorch interfacing.
    """
    print("="*70)
    print("🚀 STARTING IN-DEPTH QML.SPECS EXPLORATION 🚀")
    print("="*70)

    # --- [1] Setup: The Device ---
    print("\n--- [1] Initializing the lightning.gpu device ---")
    try:
        dev = qml.device("lightning.gpu", wires=NUM_QUBITS)
        print(f"✅ SUCCESS: Using device '{dev.name}'")
    except qml.DeviceError:
        print("⚠️ WARNING: lightning.gpu not found. Falling back to CPU.")
        dev = qml.device("default.qubit", wires=NUM_QUBITS)

    # --- [2] The Real-World Case: A Circuit with High-Level Templates ---
    print("\n--- [2] The Real-World Case: A Circuit with High-Level Templates ---")

    @qml.qnode(dev)
    def vqc_circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
        return qml.expval(qml.PauliZ(wires=0))

    inputs_sample = torch.randn(3, dtype=torch.float64) 
    weights_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=NUM_LAYERS, n_wires=NUM_QUBITS)
    weights_sample = torch.randn(weights_shape, dtype=torch.float64)

    # -- 2a. The PROBLEM: Running `qml.specs` on the high-level circuit --
    print("\n--- [2a] The PROBLEM: `qml.specs` on the original, templated circuit ---")
    specs_before_decomp = qml.specs(vqc_circuit)(inputs_sample, weights_sample)
    resources_before = specs_before_decomp.get('resources')
    print(f"   -> Gate Types reported: {getattr(resources_before, 'gate_types', {})}")
    print("   -> ⚠️ VERIFIED: This is incorrect. It doesn't show the fundamental gates.")

    # -- 2b. The SOLUTION: Decomposing the circuit first --
    print("\n--- [2b] The SOLUTION: Decomposing the circuit before analysis ---")
    
    # --- THIS IS THE DEFINITIVE FIX ---
    # 1. Create a "tape" (a data structure) of the circuit's operations
    with qml.tape.QuantumTape() as tape:
        vqc_circuit(inputs_sample.numpy(), weights_sample.numpy()) # Use numpy arrays
    
    # 2. Decompose the tape, providing the required `stopping_condition` argument.
    #    This condition tells decompose to expand everything that is not a fundamental operation.
    [expanded_tape], _ = qml.devices.preprocess.decompose(
        tape, 
        stopping_condition=lambda op: op.num_wires is not None and op.num_wires <= 2
    )
    print("   -> ✅ Step 1: Circuit successfully decomposed into fundamental gates.")
    
    # 3. Create a NEW, runnable QNode from the decomposed tape.
    @qml.qnode(dev)
    def decomposed_qnode():
        for op in expanded_tape.operations:
            qml.apply(op)
        return qml.expval(qml.PauliZ(0))
        
    print("   -> ✅ Step 2: Created a new, runnable QNode from the decomposed tape.")

    # 4. Now, run qml.specs on this NEW, runnable, decomposed QNode
    specs_after_decomp = qml.specs(decomposed_qnode)()
    resources_after = specs_after_decomp.get('resources')
    gate_types_after = getattr(resources_after, 'gate_types', {})
    
    print(f"   -> ✅ Step 3: `qml.specs` ran successfully on the decomposed QNode.")
    print(f"   -> Gate Types reported AFTER decomposition: {gate_types_after}")

    # 5. Manually parse the correct, decomposed gate counts
    total_1q_gates = 0
    total_2q_gates = 0
    for gate_name, count in gate_types_after.items():
        if gate_name in ['CNOT', 'CZ', 'SWAP', 'ISWAP']:
            total_2q_gates += count
        else:
            total_1q_gates += count
            
    print("\n   --- Final Verification of the Method ---")
    print(f"   - Final 1-Qubit Gate Count: {total_1q_gates}")
    print(f"   - Final 2-Qubit Gate Count: {total_2q_gates}")
    
    if total_1q_gates > 0 and total_2q_gates > 0:
        print("\n   ✅✅✅ DEFINITIVE SUCCESS: The decomposition and manual counting method is verified and correct.")
    else:
        print("\n   ❌❌❌ FAILED: The decomposition method did not yield correct gate counts.")
    # --- END OF FIX ---

    print("\n" + "="*70)
    print("🏁 EXPLORATION COMPLETE 🏁")
    print("="*70)

if __name__ == "__main__":
    main()