from abc import ABC, abstractmethod
import pandas as pd

class BaseAlgorithm(ABC):
    """
    An abstract base class (a blueprint) for all tracking algorithms.
    """
    @abstractmethod
    # --- REPLACE THE __init__ METHOD WITH THIS ---
    def __init__(self, config: dict):
        self.config = config
        
        # Create a "sanitized" version of the config for logging to avoid printing huge dataframes.
        config_for_log = {}
        for key, value in self.config.items():
            if isinstance(value, pd.DataFrame):
                config_for_log[key] = f"<DataFrame shape={value.shape}>"
            else:
                config_for_log[key] = value
                
        print(f"  -> Initializing {self.__class__.__name__} with config: {config_for_log}")

    @abstractmethod
    def benchmark(self, event_data: dict) -> dict:
        pass