import joblib
from sklearn.linear_model import LogisticRegression

def load_processed():
    X_train, y_train = joblib.load("data/processed/train.joblib")
    return X_train, y_train

def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=500,
        solver="saga",
        class_weight="balanced",
        n_jobs=None
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    X_train, y_train = load_processed()
    model = train_model(X_train, y_train)
    joblib.dump(model, "models/artifacts/fake_jobs_model.joblib")
    print("Training complete. Model saved to models/artifacts/fake_jobs_model.joblib")
