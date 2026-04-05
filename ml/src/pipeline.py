# ==================================
# Pipeline Module
# ==================================

# basic pipeline

import pandas as pd

# import modules
from ingestion import ingest_data
from preprocessing.text_cleaning import TextCleaner
from preprocessing import TextTokenizer
from features import FeatureEngineer

# Pipeline Function (runs whole pipeline end-to-end)
def run_pipeline(file_path: str):
    # Step 1: Ingest data
    df = ingest_data(file_path)

    # Step 2: Clean text data
        # TODO: add data cleaning module wrapper function.

    # Step 3: Text Preprocessing (tokenization for BERT)
        # TODO: add text preprocessing module wrapper function.
    
    # Step 4: Feature Engineering (TF-IDF, BoW, label encoding)
        # TODO: add feature engineering module wrapper function.


    return df