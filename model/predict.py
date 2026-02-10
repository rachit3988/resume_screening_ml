import joblib

from utils.preprocessing import clean_text
from utils.logger import get_logger

LOGGER = get_logger("PREDICT")

MODEL_PATH = "model/artifacts/resume_classifier.pkl"
VECTORIZER_PATH = "model/artifacts/tfidf_vectorizer.pkl"


def main():
    LOGGER.info("Loading model artifacts")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Example resume (realistic test case)
    sample_resume = """
    Software Engineer with 4+ years of experience in Java, Spring Boot,
    REST APIs, Microservices, AWS, Docker, Kubernetes.
    Worked on scalable backend systems and cloud deployments.
    """

    LOGGER.info("Cleaning and vectorizing resume")
    cleaned = clean_text(sample_resume)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vectorized).max()
        LOGGER.info(f"Predicted Category: {prediction}")
        LOGGER.info(f"Confidence: {confidence:.2f}")
    else:
        LOGGER.info(f"Predicted Category: {prediction}")
        LOGGER.info("Confidence: N/A (model does not support probabilities)")


if __name__ == "__main__":
    main()
