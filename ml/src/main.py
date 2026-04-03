

class Bootstrap:
    # def __init__(self):
    #     pass

    def run():
        from src.utils.paths import RAW_FILE
        from src.ingestion import ingest_data
        from src.preprocessing.text_preprocessor import TextPreprocessor
        df = ingest_data(RAW_FILE)
        tp = TextPreprocessor(df)
        tp = tp.select_column_names(['partcondition', 'discrepancy'])
        print(tp.get_data())


if __name__ == '__main__':
    app = Bootstrap
    app.run()