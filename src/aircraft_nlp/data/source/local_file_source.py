from aircraft_nlp.data.source import DataSource

import pandas as pd
from pathlib import Path


class LocalFileSource(DataSource):
    def __init__(self, path: str):
        self.path = Path(path)
        self.ext = self.path.suffix
    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(f'{self.path} not found')
        if self.ext == '.csv':
            return pd.read_csv(self.path)
        if self.ext in ['.xls', '.xlsx']:
            return pd.read_excel(self.path)
        raise ValueError(f'Invalid file type found: {self.ext}')