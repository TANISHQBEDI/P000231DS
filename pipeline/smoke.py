from aircraft_nlp.data.source import LocalFileSource
from aircraft_nlp.data.preprocessing import preprocess_dataframe
from aircraft_nlp.data.validate import validate_raw, validate_processed
from aircraft_nlp.data.splitting import split
from aircraft_nlp.models.bert import train_with_data


def main():
    # path = "data/raw/NLP_Dataset_2026_Expanded.xlsx"
    path = "data/raw/NLP_Dataset_2026.xlsx"
    source = LocalFileSource(path)
    df = source.load()
    validate_raw(df)
    df = preprocess_dataframe(df)
    validate_processed(df)
    train_df, val_df = split(df)

    train_with_data(train_df, val_df)

    

if __name__ == "__main__":
    main()