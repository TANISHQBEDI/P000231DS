
import pandas as pd

class TextPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.__data = data
    
    def get_data(self) -> pd.DataFrame:
        return self.__data
    
    def get_column_names(self) -> list[str]:
        return list(self.__data.columns)
    
    def select_column_names(self, select_columns: list[str]) -> 'TextPreprocessor':
        self.__data = self.__data[[column for column in select_columns]]
        return self

if __name__ == '__main__':
    from src.utils.paths import RAW_FILE
    df = pd.read_csv(RAW_FILE)
    tp = TextPreprocessor(df)
    tp = tp.select_column_names(['PartCondition', 'Discrepancy'])
    print(tp.get_data())