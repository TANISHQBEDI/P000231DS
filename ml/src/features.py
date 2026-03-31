import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from ml.src.ingestion import ingest_data
from ml.src.data_cleaning.DataCleaner import DataCleaner


class FeatureEngineer:
    """
    Feature Engineering class to convert text data into numerical features
    suitable for machine learning and BERT models.
    """

    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str):

        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column

        # =========================
        # Validation
        # =========================
        if self.text_column not in self.df.columns:
            raise ValueError(f"Column '{self.text_column}' not found")

        if self.label_column not in self.df.columns:
            raise ValueError(f"Column '{self.label_column}' not found")

        # =========================
        # Handle missing values
        # =========================
        self.df[self.text_column] = self.df[self.text_column].fillna("").astype(str)
        self.df[self.label_column] = self.df[self.label_column].fillna("unknown")

        self.texts = self.df[self.text_column]
        self.labels = self.df[self.label_column]

        self.label_encoder = LabelEncoder()

    # ==================================
    # Label Encoding
    # ==================================
    def encode_labels(self):
        self.y = self.label_encoder.fit_transform(self.labels)
        return self.y

    # ==================================
    # Bag of Words
    # ==================================
    def bow_features(self, max_features=5000):
        self.bow_vectorizer = CountVectorizer(max_features=max_features)
        X = self.bow_vectorizer.fit_transform(self.texts)
        self.feature_names = self.bow_vectorizer.get_feature_names_out()
        return X

    # ==================================
    # TF-IDF Features
    # ==================================
    def tfidf_features(self, max_features=5000):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        X = self.tfidf_vectorizer.fit_transform(self.texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return X

    # ==================================
    # BERT Input
    # ==================================
    def get_bert_inputs(self):
        if not hasattr(self, "y"):
            self.encode_labels()
        return self.texts.tolist(), self.y

    # ==================================
    # Full Pipeline
    # ==================================
    def process(self, method="tfidf", max_features=5000):

        y = self.encode_labels()

        print(f"Using {method.upper()} with max_features={max_features}")

        if method == "tfidf":
            X = self.tfidf_features(max_features)

        elif method == "bow":
            X = self.bow_features(max_features)

        else:
            raise ValueError("Invalid method")

        return X, y


# ==================================
# TESTING BLOCK
# ==================================
if __name__ == "__main__":

    print("TEST RUNNING - FULL PIPELINE")

    try:
        # =========================
        # STEP 1: INGESTION
        # =========================
        df = ingest_data("ml/data/raw/NLP_Dataset_2026.xlsx")

        print("Ingestion done")

        # =========================
        # STEP 2: CLEANING
        # =========================
        cleaner = DataCleaner(df)
        df_clean = cleaner.process("discrepancy").get_data()

        print("Cleaning done")

        # =========================
        # STEP 3: FEATURE ENGINEERING
        # =========================

        # CHANGE LABEL COLUMN BASED ON YOUR DATA
        label_column = "partcondition"  

        fe = FeatureEngineer(df_clean, "discrepancy", label_column)

        X, y = fe.process(method="tfidf")

        print("\nFeature Engineering Completed")
        print("Shape:", X.shape)
        print("Labels sample:", y[:5])

        print("\nSample feature vector:", X.toarray()[0])
        print("\nFeature names:", fe.feature_names[:10])

        texts, labels = fe.get_bert_inputs()
        print("\nBERT sample:", texts[:2])

    except Exception as e:
        print("Error:", e)
