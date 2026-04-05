

class Bootstrap:
    # def __init__(self):
    #     pass

    def run():
        from src.utils.paths import RAW_FILE
        from src.ingestion import ingest_data
        from src.preprocessing.text_cleaning import TextCleaner
        df = ingest_data(RAW_FILE)
        tp = TextCleaner(df)
        tp = tp.remove_duplicates().remove_null()
        print(tp.get_data())


if __name__ == '__main__':
    app = Bootstrap
    app.run()