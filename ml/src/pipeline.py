# ==================================
# Pipeline Module
# ==================================

# basic pipeline

import pandas as pd

# import modules
from src.ingestion import ingest_data
from src.pre_processing.text_cleaning import TextCleaner
from src.preprocessing import ModernBERTTokenizer
from src.features import FeatureEngineer

from src.utils.paths import RAW_FILE

# Pipeline Function (runs whole pipeline end-to-end)
def run_pipeline(file_path: str = RAW_FILE):
    # Step 1: Ingest data
    print('-'*20)
    print(f'DATA INGESTION')
    df = ingest_data(file_path)
    print('-'*20)
    print(f'DATA CLEANING')
    text_cleaner = TextCleaner(df)
    df = text_cleaner.pipe()
    print(df)
    print('-'*20)

    # Step 3: Text Preprocessing (tokenization for BERT)
        # TODO: add text preprocessing module wrapper function.
    
    # Step 4: Feature Engineering (TF-IDF, BoW, label encoding)
        # TODO: add feature engineering module wrapper function.


    return df