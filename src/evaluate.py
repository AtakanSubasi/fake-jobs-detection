import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_artifacts():
    X_test, y_test = joblib.load("data/processed/test.joblib")
    model = joblib.load("models/artifacts/fake_jobs_model.joblib")
    return X_test, y_test, model

if __name__ == "__main__":
    X_test, y_test, model = load_artifacts()

    y_pred = model.predict(X_test)
    y_prob = getattr(model, "predict_proba", None)
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob(X_test)[:, 1])
    else:
        auc = None

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")

    # optional: save to file
    with open("models/artifacts/eval.txt", "w", encoding="utf-8") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4) + "\n")
        if auc is not None:
            f.write(f"ROC AUC: {auc:.4f}\n")
