import pandas as pd
from data_cleaning import Cleaner

class DataCleaner(Cleaner):
    """
    Concrete implementation of Cleaner for general tabular data.
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def remove_duplicates(self) -> "DataCleaner":
        self.data = self.data.drop_duplicates()
        return self

    def clean_text(self, column: str) -> "DataCleaner":
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        self.data[column] = (
            self.data[column]
            .astype(str)      # handles NaN safely
            .str.strip()
            .str.lower()
        )
        return self
    
    def process(self, text_column: str = "Description") -> "DataCleaner":
        return (
                    self.remove_duplicates()
                    .clean_text(text_column)
                )