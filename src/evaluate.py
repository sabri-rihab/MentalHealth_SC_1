import json
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from src.config import MODEL_PATH, METRICS_PATH

def evaluate_model(X_test, y_test, model_path=MODEL_PATH, metricx_path=METRICS_PATH):
    pipline = joblib.load(model_path)
    y_pred = pipline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    metricx = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'report': report,
        'confusion_matrix': metricx,
    }

    with open(metricx_path, 'w') as f:
        json.dump(metrics, f, indent=4)

        return metrics