import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """
    Feature Engineering class to convert text data into numerical features
    suitable for machine learning and BERT models.
    """

    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str):
        """
        Initialize Feature Engineer

        Parameters:
        df (pd.DataFrame): Input dataframe
        text_column (str): Column containing text data
        label_column (str): Target column for classification
        """

        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column

        # =========================
        # Validation
        # =========================
        if self.text_column not in self.df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in dataframe")

        if self.label_column not in self.df.columns:
            raise ValueError(f"Column '{self.label_column}' not found in dataframe")

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
        """
        Encode target labels into numeric form
        """
        self.y = self.label_encoder.fit_transform(self.labels)
        return self.y

    # ==================================
    # Bag of Words (BoW)
    # ==================================
    def bow_features(self, max_features=5000):
        """
        Generate Bag of Words features
        """
        self.bow_vectorizer = CountVectorizer(max_features=max_features)
        X = self.bow_vectorizer.fit_transform(self.texts)

        # Store feature names (useful later)
        self.feature_names = self.bow_vectorizer.get_feature_names_out()

        return X

    # ==================================
    # TF-IDF Features (Recommended)
    # ==================================
    def tfidf_features(self, max_features=5000):
        """
        Generate TF-IDF features
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        X = self.tfidf_vectorizer.fit_transform(self.texts)

        # Store feature names (important for XAI later)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return X

    # ==================================
    # BERT Input Preparation
    # ==================================
    def get_bert_inputs(self):
        """
        Prepare raw text and encoded labels for BERT model
        (no heavy preprocessing applied)
        """
        if not hasattr(self, "y"):
            self.encode_labels()

        return self.texts.tolist(), self.y

    # ==================================
    # Full Pipeline
    # ==================================
    def process(self, method="tfidf", max_features=5000):
        """
        Run full feature engineering pipeline

        Parameters:
        method (str): 'tfidf' or 'bow'
        max_features (int): number of features

        Returns:
        X (features), y (labels)
        """

        y = self.encode_labels()

        print(f"Using {method.upper()} with max_features={max_features}")

        if method == "tfidf":
            X = self.tfidf_features(max_features)

        elif method == "bow":
            X = self.bow_features(max_features)

        else:
            raise ValueError("Invalid method. Choose 'tfidf' or 'bow'")

        return X, y


# ==================================
# TESTING BLOCK
# ==================================
if __name__ == "__main__":
    print("TEST RUNNING")

    # Sample dataset (replace with real dataset later)
    data = {
        "discrepancy": [
            "engine failure detected",
            "cabin odor reported",
            "hydraulic leak found",
            "engine failure detected"
        ],
        "label": [
            "engine",
            "cabin",
            "hydraulic",
            "engine"
        ]
    }

    df = pd.DataFrame(data)

    # Initialize Feature Engineer
    fe = FeatureEngineer(df, text_column="discrepancy", label_column="label")

    # Generate features
    X, y = fe.process(method="tfidf")

    print("\nFeature Engineering Completed")
    print("Feature matrix shape:", X.shape)
    print("Encoded labels:", y)

    # Show sample vector
    print("\nSample feature vector:", X.toarray()[0])

    # Show feature names (top 10)
    print("\nSample feature names:", fe.feature_names[:10])

    # BERT-ready data
    texts, labels = fe.get_bert_inputs()
    print("\nBERT Input Sample:")
    print(texts[:2])
