import os
import pandas as pd

# We now use the powerful, official library functions
from trackml.dataset import load_event
from trackml.utils import add_position_quantities

class DataLoader:
    """
    A class responsible for loading and providing data from the TrackML dataset,
    leveraging the official 'trackml' library for efficiency and feature engineering.
    """

# In src/data_loader.py
# REPLACE the existing __init__ method with this one:
    def __init__(self, events_path: str):
        """
        Initializes the DataLoader.
    
        Args:
            events_path (str): The full, direct path to the directory containing the event files.
        """
        self.events_path = events_path
        # We no longer need to load detectors.csv manually.
        print(f"DataLoader initialized. Using 'trackml' library.")
        print(f"Expecting event files in: {self.events_path}")
    def get_event_with_cylindrical_coords(self, event_id: str) -> tuple | None:
        """
        Loads all data for a single event using the trackml library and adds
        cylindrical coordinate features ('r', 'phi', 'rho') to the hits DataFrame.

        This provides the data in an "ML-ready" state for feature creation.

        Args:
            event_id (str): The unique identifier for the event, e.g., 'event000001000'.

        Returns:
            A tuple containing (hits, cells, particles, truth) DataFrames,
            or None if the event cannot be loaded.
        """
        print(f"\nAttempting to load data for event: {event_id}...")
        
        try:
            # The library function needs the full prefix, including the directory
            event_prefix = os.path.join(self.events_path, event_id)
            
            hits, cells, particles, truth = load_event(event_prefix)
            
            # --- Feature Engineering Step ---
            # Add cylindrical coordinates. This is a critical step.
            print("  -> Adding cylindrical coordinates (r, phi, rho) to hits data...")
            hits = add_position_quantities(hits)
            
            print("...Event loaded and enhanced successfully.")
            return hits, cells, particles, truth

        except Exception as e:
            print(f"❌ ERROR: Could not load event '{event_id}' using the trackml library.")
            print(f"  - Details: {e}")
            return None
