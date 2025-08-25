import joblib
import sys

def load_artifacts():
    model = joblib.load("models/artifacts/fake_jobs_model.joblib")
    vectorizer = joblib.load("models/artifacts/vectorizer.joblib")
    return model, vectorizer

def predict(text):
    model, vectorizer = load_artifacts()
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "FAKE" if pred == 1 else "REAL"
    return label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_cli.py \"Your job description here\"")
        sys.exit(1)

    text_input = sys.argv[1]
    result = predict(text_input)
    print(f"Prediction: {result}")
