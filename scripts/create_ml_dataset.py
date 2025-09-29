# --- Universal Path Setup ---
import sys
import os
# This block of code is designed to solve the ModuleNotFoundError
# by dynamically adding the project's root directory to the Python path.
# This allows the script to be run from anywhere, either directly or as a subprocess.
try:
    # Get the absolute path of the directory containing the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project's root directory.
    project_root = os.path.dirname(script_dir)
    # Add the project root to the system path if it's not already there.
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    # This fallback is for interactive environments where __file__ might not be defined.
    # It assumes the current working directory is the project root.
    if '.' not in sys.path:
        sys.path.append('.')
# --- End of Universal Path Setup ---

import pandas as pd
import numpy as np
import json
from src.config_loader import load_config
from src.data_loader import DataLoader

# ... (rest of the file is unchanged)


def create_segments_optimal(hits: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """
    Creates track segments using a memory-efficient "cone search" heuristic.
    This avoids the combinatorial explosion of a full cross join.
    """
    print("  -> Pre-processing hits and truth data...")
    hits['layer_num'] = hits['volume_id'] * 100 + hits['layer_id']
    hit_truth = pd.merge(hits, truth, on='hit_id')

    layer_nums = sorted(hit_truth['layer_num'].unique())
    all_segments_list = []

    print(f"  -> Generating segments using a cone search (phi < {config.PHI_CUT}, z < {config.Z_CUT})...")
    
    for i in range(len(layer_nums) - 1):
        layer1_num = layer_nums[i]
        layer2_num = layer_nums[i+1]
        
        hits1 = hit_truth[hit_truth['layer_num'] == layer1_num].copy()
        hits2 = hit_truth[hit_truth['layer_num'] == layer2_num].copy()

        if hits1.empty or hits2.empty:
            continue

        # Create a temporary key for merging that forces a cross join,
        # but we will filter immediately after.
        hits1['key'] = 1
        hits2['key'] = 1
        
        # This merge is still potentially large, but we will filter it in chunks.
        # This is an intermediate step to our final, most optimal solution.
        # A more advanced solution would use a KD-Tree, but pandas can handle this.
        
        # Let's process hits1 in chunks to keep memory usage down.
        chunk_segments = []
        for j in range(0, len(hits1), config.CHUNK_SIZE):
            chunk1 = hits1.iloc[j:j+config.CHUNK_SIZE]
            
            # This is the temporary cross-product for the chunk
            merged = pd.merge(chunk1.add_suffix('_1'), hits2.add_suffix('_2'), how='cross')
            
            # --- THE CRITICAL FILTERING STEP ---
            # Calculate phi difference and handle wraparound
            dphi = merged['phi_2'] - merged['phi_1']
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
            
            # Calculate z difference
            dz = merged['z_2'] - merged['z_1']
            
            # Apply the geometric cuts
            cut = (np.abs(dphi) < config.PHI_CUT) & (np.abs(dz) < config.Z_CUT)
            
            chunk_segments.append(merged[cut])
            
        if chunk_segments:
            all_segments_list.extend(chunk_segments)

    if not all_segments_list:
        print("  -> Warning: No segments were generated for this event.")
        return pd.DataFrame()

    full_dataset = pd.concat(all_segments_list, ignore_index=True)
    
    print("  -> Assigning labels for all segments...")
    full_dataset['label'] = (
        (full_dataset['particle_id_1'] == full_dataset['particle_id_2']) &
        (full_dataset['particle_id_1'] != 0)
    ).astype(int)

    print("  -> Calculating final geometric features...")
    full_dataset['delta_r'] = full_dataset['r_2'] - full_dataset['r_1']
    full_dataset['delta_phi'] = np.arctan2(np.sin(full_dataset['phi_2'] - full_dataset['phi_1']), np.cos(full_dataset['phi_2'] - full_dataset['phi_1']))
    full_dataset['delta_z'] = full_dataset['z_2'] - full_dataset['z_1']
    
    final_cols = ['delta_r', 'delta_phi', 'delta_z', 'label']
    return full_dataset[final_cols]


config = load_config()
# REPLACE your existing main() function with this new version
def main():
    """
    Main function to drive the creation of the ML-ready dataset.
    
    It now includes a "smart check" to see if a valid dataset for the
    current configuration already exists, skipping the expensive data
    generation process if possible.
    """
   
    print("="*60)
    print("🚀 STARTING ML-READY DATASET CREATION (with Smart Check) 🚀")
    print("="*60)

    # --- 1. Define the parameters that create the data's "fingerprint" ---
    manifest_params = {
        "NUM_EVENTS_TO_PROCESS": config.NUM_EVENTS_TO_PROCESS,
        "Z_CUT": config.Z_CUT,
        "PHI_CUT": config.PHI_CUT,
        "CHUNK_SIZE": config.CHUNK_SIZE
    }
    manifest_path = config.ML_DATASET_PATH.replace('.csv', '.manifest.json')

    # --- 2. Perform the Smart Check ---
    if os.path.exists(manifest_path):
        print(f"-> Found an existing manifest file: '{manifest_path}'")
        with open(manifest_path, 'r') as f:
            existing_manifest = json.load(f)
        
        # Compare the existing manifest with the current configuration
        if existing_manifest == manifest_params:
            print("   ✅ Manifest matches current configuration.")
            print("   Skipping data generation as a valid dataset already exists.")
            print("="*60)
            return # Exit the script successfully
        else:
            print("   ⚠️  WARNING: Manifest does not match current configuration. Data will be regenerated.")
            print(f"      - Existing: {existing_manifest}")
            print(f"      - Current:  {manifest_params}")

    # --- 3. If the check fails or no manifest exists, run the full process ---
    print("\n-> Proceeding with dataset generation...")
    loader = DataLoader(events_path=config.RAW_EVENTS_DIR)
    all_event_ids = sorted(list(set([f.split('-')[0] for f in os.listdir(loader.events_path)])))
    events_to_process = all_event_ids[:config.NUM_EVENTS_TO_PROCESS]
    
    print(f"Found {len(all_event_ids)} events. Will process the first {len(events_to_process)}.")
    
    all_event_segments = []
    for event_id in events_to_process:
        event_data = loader.get_event_with_cylindrical_coords(event_id)
        if event_data:
            hits, _, _, truth = event_data
            event_segments = create_segments_optimal(hits, truth)
            all_event_segments.append(event_segments)
            print(f"...Finished processing event {event_id}. Found {len(event_segments)} segments.")
            
    if not all_event_segments:
        print("❌ ERROR: No segments were created. Aborting.")
        return
        
    final_ml_dataset = pd.concat(all_event_segments, ignore_index=True)
    
    # --- 4. Save the Final Dataset AND the new Manifest File ---
    print("\n--- Saving Final Dataset and Manifest ---")
    final_ml_dataset.to_csv(config.ML_DATASET_PATH, index=False)
    print(f"✅ Success! Dataset saved to '{config.ML_DATASET_PATH}'")

    with open(manifest_path, 'w') as f:
        json.dump(manifest_params, f, indent=4)
    print(f"✅ Success! Manifest saved to '{manifest_path}'")
    
    print("\n" + "="*60)
    print("🔬 VERIFYING FINAL DATASET 🔬")
    # ... (The verification part is unchanged) ...
    print(f"Total segments (rows) in final dataset: {len(final_ml_dataset)}")
    print(f"Columns in final dataset: {final_ml_dataset.columns.tolist()}")
    print("\n--- [1] Label Distribution (True vs. False Segments) ---")
    print(final_ml_dataset['label'].value_counts(normalize=True))
    print("\n--- [2] Sample of 5 TRUE segments (label=1) ---")
    print(final_ml_dataset[final_ml_dataset['label'] == 1].head())

# This part of the script is unchanged
if __name__ == "__main__":
    main()
