# ==================================
# Pipeline Module
# ==================================

import pandas as pd

from src.ingestion.ingest import ingest_data
from src.preprocessing.text_cleaning import TextCleaner
from src.model.features import FeatureEngineer
from src.utils.paths import RAW_FILE

def run_pipeline(file_path: str = RAW_FILE):
    # =========================
    # STEP 1: INGESTION
    # =========================
    print('-'*20)
    print('DATA INGESTION')
    df = ingest_data(str(RAW_FILE))

    # =========================
    # STEP 2: CLEANING
    # =========================
    print('-'*20)
    print('DATA CLEANING')
    cleaner = TextCleaner(df)
    df = cleaner.pipe()
    print(df[['discrepancy', 'discrepancy_clean']])

    # =========================
    # STEP 3: FEATURE ENGINEERING
    # =========================
    print('-'*20)
    print('FEATURE ENGINEERING')

    label_column = "partcondition"
    fe = FeatureEngineer(df, "discrepancy_clean", label_column)

    X, y = fe.process(method="tfidf")

    print('-'*20)
    print("Pipeline completed")
    print('-'*20)
    
    return X, y


    # Step 3: Text Preprocessing (tokenization for BERT)
        # TODO: add text preprocessing module wrapper function.
    
    # Step 4: Feature Engineering (TF-IDF, BoW, label encoding)
        # TODO: add feature engineering module wrapper function.


# =========================
# RUN PIPELINE
# =========================
if __name__ == "__main__":
    run_pipeline()
