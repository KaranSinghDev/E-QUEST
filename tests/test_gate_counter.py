from src.quantum_vqc import QuantumVQC
import torch

def main():
    """
    Main function to specifically and quickly test the get_gate_counts method
    after the final decomposition fix.
    """
    print("="*60)
    print("🚀 STARTING FINAL GATE COUNTER VERIFICATION 🚀")
    print("="*60)

    # 1. Define a minimal configuration needed to initialize the class
    vqc_config = {
        "dataset_path": "track_segments_ml_dataset.csv",
        "num_samples": 200, # A small number, just to initialize the data tensors
        "num_layers": 2,    # Must match the layers used in the real test
    }

    print("\n[Step 1: Initializing QuantumVQC Algorithm]")
    q_algo = QuantumVQC(config=vqc_config)
    torch.manual_seed(42)

    # 2. Prepare the necessary data
    # We MUST run _load_and_prepare_data because get_gate_counts needs the tensors it creates.
    print("\n[Step 2: Preparing data tensors]")
    q_algo._load_and_prepare_data()

    # 3. Call ONLY the function we want to test
    print("\n[Step 3: Calling the final get_gate_counts()]")
    gate_counts = q_algo.get_gate_counts()

    # 4. Verify the results
    print("\n[Step 4: Verification]")
    if gate_counts:
        n_1q = gate_counts.get("1q_gates_per_call", 0)
        n_2q = gate_counts.get("2q_gates_per_call", 0)

        print(f"  - Final 1-Qubit Gates Returned: {n_1q}")
        print(f"  - Final 2-Qubit Gates Returned: {n_2q}")
        
        # The most important check: are the counts non-zero and reasonable?
        # For a StronglyEntanglingLayers with 2 layers and 4 qubits, we expect many gates.
        if n_1q > 10 and n_2q > 5:
            print("\n✅ Verification PASSED: Gate counts are non-zero and appear correct.")
        else:
            print("\n❌ Verification FAILED: Gate counts are unexpectedly low.")
    else:
        print("❌ FAILED: The method did not return any results.")
        
    print("\n" + "="*60)
    print("🎉 FINAL GATE COUNTER VERIFICATION COMPLETE 🎉")
    print("="*60)

if __name__ == "__main__":
    main()