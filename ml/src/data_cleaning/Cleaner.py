from abc import ABC, abstractmethod
import pandas as pd


class Cleaner(ABC):
    """
    Abstract base class for all data cleaners.
    Defines the cleaning pipeline (Template Method Pattern).
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    @abstractmethod
    def remove_duplicates(self) -> "Cleaner":
        pass

    @abstractmethod
    def clean_text(self, column: str) -> "Cleaner":
        pass
    
    @abstractmethod
    def process(self, text_column: str) -> "Cleaner":
        pass

    def get_data(self) -> pd.DataFrame:
        return self.data