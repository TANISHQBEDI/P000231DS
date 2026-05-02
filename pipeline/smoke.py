from aircraft_nlp.data.source import LocalFileSource

def main():
    path = 'data/raw/NLP_Dataset_2026_Expanded.xlsx'
    source = LocalFileSource(path)
    df = source.load()
    print(df.info())

if __name__ == '__main__':
    main()