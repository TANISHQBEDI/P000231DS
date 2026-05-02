from aircraft_nlp.data.source import LocalFileSource
from aircraft_nlp.data.preprocessing import preprocess_dataframe

from aircraft_nlp.data.validate import validate_raw, validate_processed

def main():
    path = "data/raw/NLP_Dataset_2026_Expanded.xlsx"
    source = LocalFileSource(path)
    df = source.load()
    validate_raw(df)
    df = preprocess_dataframe(df)
    validate_processed(df)



if __name__ == "__main__":
    main()