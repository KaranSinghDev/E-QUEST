from abc import ABC, abstractmethod
import pandas as pd

class BaseAlgorithm(ABC):
    """
    An abstract base class (a blueprint) for all tracking algorithms.
    """
    @abstractmethod
    def __init__(self, config: dict):
        self.config = config
        print(f"  -> Initializing {self.__class__.__name__} with config: {self.config}")

    @abstractmethod
    def benchmark(self, event_data: dict) -> dict:
        pass