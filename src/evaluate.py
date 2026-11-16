import json
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.config import MODEL_PATH, METRICS_PATH

def evaluate_model(model_path = MODEL_PATH, X_test=None, y_test=None, metricx_path=METRICS_PATH):
    pipline = joblib.load(model_path)
    y_pred = pipline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    metricx = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_metrix' : metricx
    }

    with open(metricx_path, 'w') as f:
        json.dump(metrics, f, indent=4)

        return metrics