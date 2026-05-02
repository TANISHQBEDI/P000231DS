from aircraft_nlp.data.source import DataSource

class S3DataSource(DataSource):
    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key