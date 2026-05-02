from aircraft_nlp.data.source import LocalFileSource
from aircraft_nlp.data.preprocessing import normalize_text

def main():
    path = "data/raw/NLP_Dataset_2026_Expanded.xlsx"
    source = LocalFileSource(path)
    df = source.load()

    # Pick your text column name
    text_col = "Discrepancy"

    df[text_col] = df[text_col].fillna("").astype(str).apply(normalize_text)
    print(df['Discrepancy'].head(20))

if __name__ == "__main__":
    main()