# ==================================
# Feature Engineering Module
# ==================================

# NOTES
# - "This module converts cleaned text data into numerical features for machine learning models (TF-IDF, BoW) and prepares inputs for BERT".

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Import previous pipeline modules
from ml.src.ingestion import ingest_data
from ml.src.preprocessing.text_cleaning import TextCleaner as DataCleaner


class FeatureEngineer:
    """
    This class performs feature engineering on text data.
    It converts text into numerical representations and encodes labels.
    """

    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str):
        """
        Initialize the FeatureEngineer with dataset and column names.

        Parameters:
        df: cleaned dataframe from preprocessing
        text_column: column containing text data (e.g., 'discrepancy')
        label_column: target column for classification
        """

        # Create a copy of the dataset to avoid modifying original data
        self.df = df.copy()

        self.text_column = text_column
        self.label_column = label_column

        # =========================
        # Validation checks
        # =========================
        # Ensure required columns exist
        if self.text_column not in self.df.columns:
            raise ValueError(f"Column '{self.text_column}' not found")

        if self.label_column not in self.df.columns:
            raise ValueError(f"Column '{self.label_column}' not found")

        # =========================
        # Handle missing values
        # =========================
        # Replace missing text with empty string and ensure string type
        self.df[self.text_column] = self.df[self.text_column].fillna("").astype(str)

        # Replace missing labels with 'unknown'
        self.df[self.label_column] = self.df[self.label_column].fillna("unknown")

        # Extract text and labels
        self.texts = self.df[self.text_column]
        self.labels = self.df[self.label_column]

        # Initialize label encoder
        self.label_encoder = LabelEncoder()

    # ==================================
    # Label Encoding
    # ==================================
    def encode_labels(self):
        """
        Convert categorical labels into numeric format.
        """
        self.y = self.label_encoder.fit_transform(self.labels)
        return self.y

    # ==================================
    # Bag of Words (BoW)
    # ==================================
    def bow_features(self, max_features=5000):
        """
        Convert text into Bag-of-Words representation.
        Each word is represented by its frequency.
        """
        self.bow_vectorizer = CountVectorizer(max_features=max_features)

        # Transform text into numerical matrix
        X = self.bow_vectorizer.fit_transform(self.texts)

        # Store feature names for interpretation
        self.feature_names = self.bow_vectorizer.get_feature_names_out()

        return X

    # ==================================
    # TF-IDF Features (Primary Method)
    # ==================================
    def tfidf_features(self, max_features=5000):
        """
        Convert text into TF-IDF representation.
        Captures importance of words relative to document frequency.
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

        # Transform text into TF-IDF matrix
        X = self.tfidf_vectorizer.fit_transform(self.texts)

        # Store feature names (useful for explainability later)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return X

    # ==================================
    # BERT Input Preparation
    # ==================================
    def get_bert_inputs(self):
        """
        Prepare raw text and labels for BERT model.
        Note: No heavy preprocessing is applied to preserve context.
        """
        if not hasattr(self, "y"):
            self.encode_labels()

        return self.texts.tolist(), self.y

    # ==================================
    # Full Feature Engineering Pipeline
    # ==================================
    def process(self, method="tfidf", max_features=5000):
        """
        Run complete feature engineering pipeline.

        Parameters:
        method: 'tfidf' or 'bow'
        max_features: number of features to keep

        Returns:
        X: feature matrix
        y: encoded labels
        """

        # Encode labels
        y = self.encode_labels()

        print(f"Using {method.upper()} with max_features={max_features}")

        # Generate features based on selected method
        if method == "tfidf":
            X = self.tfidf_features(max_features)

        elif method == "bow":
            X = self.bow_features(max_features)

        else:
            raise ValueError("Invalid method")

        return X, y


# ==================================
# TESTING BLOCK (FULL PIPELINE)
# ==================================
if __name__ == "__main__":

    print("TEST RUNNING - FULL PIPELINE")

    try:
        # =========================
        # STEP 1: INGESTION
        # =========================
        df = ingest_data("ml/data/raw/NLP_Dataset_2026.xlsx")
        print("Ingestion completed")

        # =========================
        # STEP 2: DATA CLEANING
        # =========================
        cleaner = DataCleaner(df)
        df_clean = cleaner.process("discrepancy").get_data()
        print("Cleaning completed")

        # =========================
        # STEP 3: FEATURE ENGINEERING
        # =========================
        # NOTE: Update label_column based on dataset
        label_column = "partcondition"

        fe = FeatureEngineer(df_clean, "discrepancy", label_column)

        X, y = fe.process(method="tfidf")

        print("\nFeature Engineering Completed")
        print("Shape:", X.shape)
        print("Labels sample:", y[:5])

        # Display sample feature vector
        print("\nSample feature vector:", X.toarray()[0])

        # Display some feature names
        print("\nFeature names:", fe.feature_names[:10])

        # BERT-ready inputs
        texts, labels = fe.get_bert_inputs()
        print("\nBERT sample:", texts[:2])

    except Exception as e:
        print("Error:", e)
