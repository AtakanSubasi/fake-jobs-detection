# Fake Job Postings Detection

End-to-end machine learning pipeline to detect **fake job postings**.

## 🚀 Project Overview
- Dataset: [Fake Job Postings Dataset (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Goal: Classify job postings as **REAL** or **FAKE** using NLP and ML
- Stack: Python, Pandas, Scikit-learn, NLTK, Joblib

⚠️ Note: The dataset file (`fake_job_postings.csv`) is **not included** in this repository due to size restrictions.  
Download it from Kaggle and place it under:
data/raw/fake_job_postings.csv

## 📂 Project Structure
fake-jobs-detection/
* data/
  * raw/          # raw CSV (not uploaded to GitHub)
  * interim/      # cleaned data (ignored in git)
  * processed/    # train/test splits (ignored in git)
* models/
  * artifacts/    # trained model & vectorizer (ignored in git)
* src/
  * preprocess.py # clean & vectorize
  * train.py      # train model
  * evaluate.py   # metrics
  * predict_cli.py # CLI predictions
* requirements.txt
* README.md

## ⚡ How to Run

# 1. Preprocess data
python src/preprocess.py

# 2. Train model
python src/train.py

# 3. Evaluate model
python src/evaluate.py

# 4. Predict CLI
python src/predict_cli.py "Looking for a Data Scientist with Python and SQL experience"


## 📊 Results
- Accuracy: ~88%
- ROC AUC: ~0.98
- High recall for fraud detection (important to catch fake jobs)

## 🔮 Next Steps
- Try other models (Naive Bayes, RandomForest, XGBoost)
- Hyperparameter tuning
- Deploy as API (FastAPI/Flask)
- Add visualization dashboard





