# ==================================
# Feature Engineering Module
# ==================================

# NOTES
# - Converts cleaned text into numerical features (TF-IDF, BoW)
# - Prepares inputs for BERT
# - Includes merging of part_name + discrepancy for richer context

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Import previous pipeline modules
from src.ingestion.ingest import ingest_data
from src.preprocessing.text_cleaning import TextCleaner as DataCleaner


class FeatureEngineer:
    """
    Performs feature engineering on text data.
    Converts text into numerical representations and encodes labels.
    """

    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str):
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

        #===========================
        # Merge part_name + discrepancy
        #===========================
        if "part_name" in self.df.columns:
            self.df["part_name"] = self.df["part_name"].fillna("").astype(str)

            self.df["combined_text"] = (
                "Part: " + self.df["part_name"] +
                " | Issue: " + self.df[self.text_column]
            )

            self.texts = self.df["combined_text"]

        else:
            # fallback if part_name not available
            self.texts = self.df[self.text_column]
    
        # Labels
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
    # Full Pipeline
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



