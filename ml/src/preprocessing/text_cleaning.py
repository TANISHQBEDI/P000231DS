
import pandas as pd

class TextCleaner:
    def __init__(self, data: pd.DataFrame):
        self.__data = data
    
    def get_data(self) -> pd.DataFrame:
        return self.__data
    
    def remove_duplicates(self, columns: list[str] = ['discrepancy']) -> 'TextCleaner':
        self.__data = self.__data.drop_duplicates(subset=[*columns])
        return self

    def remove_null(self, columns: list[str] = ['discrepancy']) -> 'TextCleaner':
        self.__data = self.__data.dropna(subset=[*columns])
        return self


if __name__ == '__main__':
    from src.utils.paths import RAW_FILE
    df = pd.read_csv(RAW_FILE)
    tp = TextCleaner(df)
    tp = tp.remove_duplicates().remove_null()