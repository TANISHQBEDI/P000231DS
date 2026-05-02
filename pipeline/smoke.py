from aircraft_nlp.data.source import LocalFileSource
from aircraft_nlp.data.preprocessing import preprocess_dataframe

def main():
    path = "data/raw/NLP_Dataset_2026_Expanded.xlsx"
    source = LocalFileSource(path)
    df = source.load()

    df = preprocess_dataframe(df)
    print(df['text'].head(20))

if __name__ == "__main__":
    main()