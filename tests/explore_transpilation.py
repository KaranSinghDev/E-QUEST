import pennylane as qml
from pennylane import numpy as np

def analyze_circuit(qnode, title, params):
    """
    A helper function to draw a circuit and programmatically count its gates
    using the most robust 'tape' method.
    """
    print("\n" + "="*60)
    print(f"ANALYSIS OF: {title}")
    print("="*60)

    # 1. Draw the circuit diagram (this part was already working)
    print("\n[--- Circuit Diagram ---]")
    print(qml.draw(qnode)(params))

    # 2. Get the gate counts using the robust tape inspection method
    print("\n[--- Gate Counts (from Tape Inspection) ---]")
    
    # --- THIS IS THE CORRECTED, ROBUST METHOD ---
    with qml.tape.QuantumTape() as tape:
        qnode(params)
    
    gate_types = {}
    for op in tape.operations:
        gate_types[op.name] = gate_types.get(op.name, 0) + 1
    # --- END OF CORRECTION ---

    if not gate_types:
        print("No gates found.")
    else:
        for gate, count in gate_types.items():
            print(f"  - {gate}: {count}")
    
    return gate_types

def main():
    """Main function to drive the transpilation test."""
    print("#"*60)
    print("🚀 PENNYLANE TRANSPILATION VERIFICATION SCRIPT (Corrected) 🚀")
    print("#"*60)
    
    dev = qml.device("default.qubit", wires=3)

    # 1. Define our ORIGINAL circuit with high-level gates
    @qml.qnode(dev)
    def original_circuit(params):
        qml.Rot(params[0], params[1], params[2], wires=0)
        qml.Hadamard(wires=1)
        qml.Toffoli(wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(0))

    dummy_params = np.array([0.5, 0.2, 0.9], requires_grad=False)
    original_gates = analyze_circuit(original_circuit, "Original High-Level Circuit", dummy_params)

    # 2. Define our target basis set and COMPILE the circuit
    print("\n" + "#"*60)
    print("⚙️  PERFORMING TRANSPILATION...")
    print("#"*60)
    
    target_basis_set = ["RX", "RY", "RZ", "CNOT"]
    print(f"\nTarget basis gate set: {target_basis_set}")

    transpiled_circuit = qml.compile(
        original_circuit,
        basis_set=target_basis_set,
        num_passes=2
    )
    print("\n✅ Transpilation complete.")

    # 3. Analyze the "AFTER" state of the transpiled circuit
    transpiled_gates = analyze_circuit(transpiled_circuit, "Transpiled Low-Level Circuit", dummy_params)

    # 4. FINAL VERDICT: Programmatically check the results
    print("\n" + "#"*60)
    print("🔬 FINAL VERDICT 🔬")
    print("#"*60)

    disappeared_gates = set(original_gates.keys()) - set(transpiled_gates.keys())
    print(f"\nHigh-level gates removed: {disappeared_gates}")

    new_gate_set = set(transpiled_gates.keys())
    is_fully_decomposed = new_gate_set.issubset(set(target_basis_set))

    print(f"Gates in final circuit: {new_gate_set}")
    print(f"Are all final gates in target set? -> {is_fully_decomposed}")

    if is_fully_decomposed and 'Toffoli' in disappeared_gates and 'Rot' in disappeared_gates:
        print("\n\033[92m" + "SUCCESS: PennyLane correctly transpiled the circuit to the target basis set." + "\033[0m")
    else:
        print("\n\033[91m" + "FAILURE: Transpilation was not successful or complete." + "\033[0m")

if __name__ == "__main__":
    main()