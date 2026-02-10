import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.data_loader import load_and_clean_data
from utils.preprocessing import clean_text
from utils.logger import get_logger

LOGGER = get_logger("TRAIN")

DATA_PATH = "data/resume_dataset.csv"
MODEL_PATH = "model/artifacts/resume_classifier.pkl"
VECTORIZER_PATH = "model/artifacts/tfidf_vectorizer.pkl"

def main():
    from utils.data_loader import load_and_clean_data

    df = load_and_clean_data(DATA_PATH, min_samples_per_class=5)

    X = df["cleaned_resume"]
    y = df["Category"]


    MIN_SAMPLES_PER_CLASS = 5  # production-safe threshold

    class_counts = df["Category"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index

    df = df[df["Category"].isin(valid_classes)]

    LOGGER.info(
        f"After removing rare classes (<{MIN_SAMPLES_PER_CLASS} samples): {df.shape}"
    )
    LOGGER.info(
        f"Remaining classes: {df['Category'].nunique()}"
    )


    X = df["cleaned_resume"]
    y = df["Category"]

    LOGGER.info("Splitting data")
    LOGGER.info("Checking for NaNs")
    LOGGER.info(df.isna().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    LOGGER.info("TF-IDF vectorization")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    LOGGER.info("Training Logistic Regression model")
    model = LogisticRegression(
       max_iter=2000,
        class_weight="balanced",
        solver="saga"
    )
    model.fit(X_train_tfidf, y_train)

    LOGGER.info("Saving model artifacts")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    LOGGER.info("Training complete")

if __name__ == "__main__":
    main()
