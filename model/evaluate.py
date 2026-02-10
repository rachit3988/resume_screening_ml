import joblib
from sklearn.model_selection import train_test_split

from utils.data_loader import load_and_clean_data
from utils.metrics import evaluate_metrics
from utils.logger import get_logger

LOGGER = get_logger("EVAL")

DATA_PATH = "data/resume_dataset.csv"
MODEL_PATH = "model/artifacts/resume_classifier.pkl"
VECTORIZER_PATH = "model/artifacts/tfidf_vectorizer.pkl"

def main():
    df = load_and_clean_data(DATA_PATH, min_samples_per_class=5)

    X = df["cleaned_resume"]
    y = df["Category"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    metrics = evaluate_metrics(y_test, y_pred)

    LOGGER.info("Evaluation Results")
    for k, v in metrics.items():
        LOGGER.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
