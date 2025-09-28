import pennylane as qml
import torch

def run_diagnostics():
    """
    Runs a series of targeted diagnostics to isolate the root cause of the
    'QuantumTape' object has no attribute 'device_options' error.
    """
    print("="*70)
    print("🚀 STARTING PENNYLANE TAPE & DEVICE DIAGNOSTIC SCRIPT 🚀")
    print("="*70)

    # --- Test 1: Manually Inspect a QuantumTape Object ---
    print("\n--- [1] Manually creating and inspecting a QuantumTape ---")
    
    # We will build a simple circuit's tape manually to see what attributes it has.
    with qml.tape.QuantumTape() as tape:
        qml.RX(0.5, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))
    
    print(f"   -> Successfully created a QuantumTape object: {type(tape)}")
    
    # This is the most direct test of the error message.
    has_device_options = hasattr(tape, 'device_options')
    
    print(f"   -> Does this tape have a '.device_options' attribute? --- > {has_device_options}")
    if not has_device_options:
        print("   ✅ VERIFIED: Standard QuantumTapes do NOT have '.device_options'.")
        print("      This suggests the error comes from how the device processes the tape.")
    else:
        print("   ⚠️ UNEXPECTED: This tape has '.device_options', which is unusual.")
    
    # --- Test 2: Test `qml.specs` with the CPU Simulator ---
    print("\n--- [2] Testing `qml.specs` with the CPU simulator ('default.qubit') ---")
    try:
        dev_cpu = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev_cpu)
        def cpu_circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(0))
            
        # The critical test: Does qml.specs work with the CPU device?
        specs_result = qml.specs(cpu_circuit)(0.5)
        
        print("✅ SUCCESS: `qml.specs` ran without error on the 'default.qubit' device.")
        print(f"   -> CPU specs output: {specs_result}")

    except AttributeError as e:
        print(f"❌ FAILURE: The AttributeError ALSO occurs on the CPU device.")
        print(f"   -> Error: {e}")
        print("   -> This would suggest the problem is within the main PennyLane library.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during the CPU test: {e}")

    # --- Test 3: Test `qml.specs` with the GPU Simulator ---
    print("\n--- [3] Testing `qml.specs` with the GPU simulator ('lightning.gpu') ---")
    try:
        dev_gpu = qml.device("lightning.gpu", wires=2)
        
        @qml.qnode(dev_gpu)
        def gpu_circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(0))

        # The critical test: Does qml.specs work with the GPU device?
        specs_result = qml.specs(gpu_circuit)(0.5)
        
        print("✅ SUCCESS: `qml.specs` also ran without error on the 'lightning.gpu' device.")
        print(f"   -> GPU specs output: {specs_result}")

    except AttributeError as e:
        print(f"❌ FAILURE: The AttributeError occurs on the GPU device.")
        print(f"   -> Error: {e}")
        print("   -> This strongly suggests the problem is specifically within the 'pennylane-lightning-gpu' plugin.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during the GPU test: {e}")
        
    print("\n" + "="*70)
    print("🏁 DIAGNOSTIC COMPLETE 🏁")
    print("="*70)

if __name__ == "__main__":
    run_diagnostics()