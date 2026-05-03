# ==================================
# Ingestion Module 
# Mitchell Hughes
# 31/03/2026
# Notes:
# ================================== 
# - This module is responsible for ingesting data from various sources, validating it, and standardising it for use in the ML pipeline.
# - The main function is `ingest_data` (line 157), which will run through the entire ingestion process.
# - The output will be the cleaned dataframe, and will also save a copy of the raw data for traceability.s

import pandas as pd
from datetime import datetime
import os
from src.utils.paths import RAW_DIR, RAW_FILE


# ==================================
# Configuration
# ==================================

# Allowed file extensions for ingestion
ALLOWED_EXTENSIONS = ['.csv', '.xlsx']

# TODO: MAY NEED CHANGES
REQUIRED_COLUMNS = ['OperatorControlNumber','Discrepancy']


# ==================================
# Main Ingestion Function
# ==================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file or Excel file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV or Excel file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """

    # Get the file extension and validate it
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}. Allowed extensions are: {ALLOWED_EXTENSIONS}")

    try:
        if ext == '.csv':
            data = pd.read_csv(file_path)

        elif ext == '.xlsx':
            data = pd.read_excel(file_path)

        return data
    
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {str(e)}")


# ==================================
# Validation Function
# ==================================

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the DataFrame to ensure it contains the required columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate.

    Raises:
    ValueError: If any required columns are missing.
    """

    # Check if the Data is empty
    if df.empty or df is None:
        raise ValueError("The Data is empty. No data to validate.")

    # Check for missing required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Nulls are handled in ingest_data by dropping invalid rows.
    

# ==================================
# Standardisation Function
# ==================================

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to a consistent format.

    Parameters:
    df (pd.DataFrame): The DataFrame with original column names.

    Returns:
    pd.DataFrame: The DataFrame with standardised column names.
    """

    # Standardise column names by stripping whitespace, converting to lowercase, and replacing spaces with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Remove any empty rows
    df = df.dropna(how='all')

    return df


# ==================================
# Save Raw Data
# ==================================

def save_raw_copy(df: pd.DataFrame, original_filename: str) -> str:
    """
    Saves a timestamped copy of ingested data into /data/raw/
    """

    os.makedirs(RAW_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{os.path.basename(original_filename)}"

    save_path = RAW_DIR / filename

    df.to_csv(save_path, index=False)

    return save_path


# ==================================
# Pipeline Wrapper Function
# ==================================

def ingest_data(file_path: str) -> pd.DataFrame:
    """
    Ingest data from a file, validate it, standardise it, and save the raw data.

    Parameters:
    file_path (str): The path to the input file.

    Returns:
    pd.DataFrame: The validated and standardised DataFrame.
    """

    # Load the data
    df = load_data(file_path)

    # Validate basic structure and required columns
    validate_data(df)

    # Keep only rows that contain all required values.
    # This prevents a full pipeline failure when only a subset is invalid.
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError(
            "Some required columns contain null values. No valid data to process."
        )

    # Standardise column names
    df = standardise_columns(df)

    # Save the raw data
    save_raw_copy(df, original_filename=os.path.basename(file_path))

    return df


# ==================================
# Tester
# ==================================

if __name__ == "__main__":

    # TODO: Add testing code here to run the ingestion pipeline on a sample file and print the resulting DataFrame.

    # Test file path
    sample_file = RAW_FILE

    try:
        df = ingest_data(sample_file)
        print(df.head())
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")

    