from sklearn.model_selection import train_test_split
import pandas as pd

def split(df: pd.DataFrame):
    train_df, val_df = train_test_split(df, train_size=.7, stratify=df['label'])
    return train_df, val_df