import pandas as pd
from utils.preprocessing import clean_text

def load_and_clean_data(
    data_path: str,
    min_samples_per_class: int = 5
):
    df = pd.read_csv(data_path)

    # Keep only required columns
    df = df[["Resume", "Category"]]

    # Drop missing values
    df = df.dropna(subset=["Resume", "Category"])

    # Normalize labels
    df["Category"] = df["Category"].str.strip().str.upper()

    # Remove obviously corrupted labels
    df = df[df["Category"].str.len() < 40]
    df = df[df["Category"].str.contains(r"^[A-Z\- ]+$", regex=True)]

    # Clean resume text
    df["cleaned_resume"] = df["Resume"].apply(clean_text)

    # Drop empty resumes
    df = df[df["cleaned_resume"].str.strip() != ""]

    # Remove rare classes
    class_counts = df["Category"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df["Category"].isin(valid_classes)]

    return df
