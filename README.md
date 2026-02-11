# Resume Screening Machine Learning System

An end-to-end machine learning system that automatically classifies resumes into job categories using classical NLP techniques.  
The project focuses on **robust data preprocessing**, **reproducible training**, and **production-ready inference**, rather than UI or deployment.

---

## Problem Statement

Recruiters often receive hundreds or thousands of resumes for a single role.  
Manual screening is time-consuming, inconsistent, and error-prone.

This project automates resume categorization by training a machine learning model on resume text, enabling faster and more consistent candidate shortlisting.

---

## Solution Overview

The system uses:
- Text preprocessing and normalization
- TF-IDF feature extraction
- Multi-class Logistic Regression for classification
- Deterministic training and evaluation pipelines
- Serialized model artifacts for reuse during inference

The focus is on **real-world data handling**, not toy examples.

---

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF (unigrams + bigrams)
- **Model Persistence:** Joblib
- **Logging:** Python logging module

---

## Dataset

- Open-source resume dataset containing text-based resumes and job categories
- Dataset required significant cleaning due to:
  - Corrupted labels
  - HTML artifacts inside fields
  - Missing values
  - Extremely rare classes

### Dataset Cleaning Steps
- Retained only relevant columns (`Resume`, `Category`)
- Dropped rows with missing resumes or labels
- Removed empty resumes after text cleaning
- Filtered out job categories with fewer than 5 samples
- Normalized category labels

---

## Project Structure

```
resume_screening_ml/
│
├── data/
│ └── resume_dataset.csv
│
├── model/
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ ├── artifacts/
│ │ ├── resume_classifier.pkl
│ │ └── tfidf_vectorizer.pkl
│ └── init.py
│
├── utils/
│ ├── preprocessing.py
│ ├── data_loader.py
│ ├── metrics.py
│ ├── logger.py
│ └── init.py
│
├── requirements.txt
└── README.md
```


---

## Machine Learning Pipeline

1. Load and validate dataset
2. Clean and normalize resume text
3. Remove invalid and rare classes
4. Perform stratified train-test split
5. Convert text to numerical features using TF-IDF
6. Train multi-class Logistic Regression model
7. Evaluate using accuracy, precision, recall, and F1-score
8. Save trained model and vectorizer
9. Run inference on unseen resume text

---

## Model Details

- **Algorithm:** Logistic Regression (multi-class)
- **Class Handling:** `class_weight="balanced"`
- **Feature Extraction:** TF-IDF with unigrams and bigrams
- **Evaluation Strategy:** Stratified hold-out test set

---

## Evaluation Results

| Metric     | Score |
|-----------|-------|
| Accuracy  | ~71%  |
| Precision| ~73%  |
| Recall   | ~71%  |
| F1-score | ~70%  |

> These results represent a strong classical NLP baseline on noisy, real-world resume data with 25 job categories.

---

## How to Run the Project

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
```python -m model.train```

### 3. Evaluate the Model
```python -m model.evaluate```

### 4. Run Inference
```python -m model.predict```

## Key Engineering Decisions

- Used TF-IDF + Logistic Regression for interpretability and fast inference
- Centralized preprocessing logic to avoid train/evaluation skew
- Removed rare classes to ensure stable stratified splitting
- Avoided UI and deployment to focus on ML robustness and correctness
- Ensured deterministic and reproducible training runs

## Limitations

- Minority classes may receive fewer predictions due to class imbalance
- TF-IDF does not capture semantic meaning beyond token frequency
- Performance can be improved with richer text representations

## Future Improvements

- Replace TF-IDF with transformer-based embeddings (e.g., SBERT)
- Improve minority class performance using data balancing
- Expose inference via a FastAPI service

## Author

### Rachit Sharma
### Machine Learning / Software Engineer
