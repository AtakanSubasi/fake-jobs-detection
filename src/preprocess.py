import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path="data/raw/fake_job_postings.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Drop rows with too many missing values
    df = df.dropna(subset=["title", "description"])

    # Keep only useful columns
    text_columns = ["title", "location", "department", "company_profile",
                    "description", "requirements", "benefits"]
    df["text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

    # Target variable
    y = df["fraudulent"]

    return df["text"], y

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save vectorizer for later use
    joblib.dump(vectorizer, "models/artifacts/vectorizer.joblib")

    return X_train_tfidf, X_test_tfidf

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    # Save processed splits
    joblib.dump((X_train_tfidf, y_train), "data/processed/train.joblib")
    joblib.dump((X_test_tfidf, y_test), "data/processed/test.joblib")

    print("Preprocessing complete. Files saved in data/processed/")
