# --- Universal Path Setup ---
import sys
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    if '.' not in sys.path:
        sys.path.append('.')
# --- End of Universal Path Setup ---

# ✅ NOW import the runtime config safely
try:
    from src.config import DATA_PREPROCESSING_MODE
except Exception as e:
    print("❌ Could not import DATA_PREPROCESSING_MODE from src.config")
    print("   Did run_benchmark.py create config.py?")
    print(f"   Error: {e}")
    sys.exit(1)

import pandas as pd
import sys


def main():
    """
    This script is a dedicated pre-processing step in the E-QUEST workflow.
    Its primary purpose is to address severe class imbalance in the dataset
    by creating a new, perfectly balanced dataset using an undersampling technique.

    Workflow:
    1. Receives the path to a large, imbalanced dataset from the conductor script.
    2. Separates the data into majority (label=0) and minority (label=1) classes.
    3. Randomly discards a large portion of the majority class samples until
       the number of majority samples equals the number of minority samples.
    4. Combines the downsampled majority class with the full minority class.
    5. Saves this new, smaller, balanced dataset to a new file.
    6. Communicates the path of this new balanced file back to the conductor.
    """
    print("="*60)
    print("🚀 STARTING DATASET PRE-PROCESSING (BALANCING) 🚀")
    print("="*60)

    # --- Step 1: Get the input file path from the conductor ---
    if len(sys.argv) < 2:
        print("❌ CRITICAL ERROR: This script must be called with the path to the input dataset.")
        print("   It is intended to be run by the main 'run_benchmark.py' script.")
        sys.exit(1)
    
    imbalanced_dataset_path = sys.argv[1]
    print(f"-> Loading imbalanced dataset from: '{imbalanced_dataset_path}'")

    try:
        df = pd.read_csv(imbalanced_dataset_path)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: File not found at '{imbalanced_dataset_path}'.")
        print("   Ensure the 'create_ml_dataset.py' script ran successfully.")
        sys.exit(1)

    print(f"  -> Original imbalanced dataset contains {len(df):,} rows.")

    # ============================================================
    # NEW: Conditional behavior based on preprocessing mode
    # ============================================================

    print(f"  -> Original imbalanced dataset contains {len(df):,} rows.")
    print(f"  -> Selected preprocessing mode: {DATA_PREPROCESSING_MODE}")

    if DATA_PREPROCESSING_MODE == "undersample":
        print("  -> Performing undersampling to create a 1:1 balanced dataset...")

        majority_class_df = df[df['label'] == 0]
        minority_class_df = df[df['label'] == 1]

        num_minority_samples = len(minority_class_df)

        if num_minority_samples == 0:
            print("❌ CRITICAL ERROR: No positive samples (label=1) found in the dataset. Cannot balance.")
            sys.exit(1)
        
        print(f"  -> Found {num_minority_samples:,} positive ('true') samples.")
        print(f"  -> Downsampling the negative ('false') class to match this count...")

        majority_downsampled_df = majority_class_df.sample(
            n=num_minority_samples,
            random_state=42
        )

        balanced_df = pd.concat([majority_downsampled_df, minority_class_df])
        balanced_df = balanced_df.sample(frac=1, random_state=42)

        print(f"  -> New balanced dataset has {len(balanced_df):,} rows in total.")
        balanced_dataset_path = imbalanced_dataset_path.replace('.csv', '_balanced.csv')

        print(f"\n-> Saving balanced dataset to: '{balanced_dataset_path}'")
        balanced_df.to_csv(balanced_dataset_path, index=False)
        final_dataset_path = balanced_dataset_path
        print("✅ Balanced dataset saved successfully.")

    else:
        # ======================================================
        # WEIGHTED BCE MODE → DO NOT MODIFY DATASET
        # ======================================================
        print("  -> Weighted-BCE mode selected.")
        print("  -> Skipping undersampling. Keeping FULL dataset as-is.")
        final_dataset_path = imbalanced_dataset_path

    # --- CRITICAL: Tell conductor what dataset to use next ---
    with open("current_dataset_path.txt", "w") as f:
        f.write(final_dataset_path)

    print(f"  -> The workflow will now proceed using: {final_dataset_path}")


    print("\n✅ Pre-processing complete.")
    print("="*60)

if __name__ == "__main__":
    main()