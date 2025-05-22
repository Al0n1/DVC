import numpy as np
import joblib
import os
import dvclive
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_model_testing(base_path: str):
    models_dir = os.path.join(base_path, 'models')
    data_processed_dir = os.path.join(base_path, 'data', 'processed')
    dvclive_dir = os.path.join(base_path, 'dvclive')

    os.makedirs(dvclive_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'rf_model.joblib')
    model = joblib.load(model_path)

    X_test = np.load(os.path.join(data_processed_dir, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_processed_dir, 'y_test.npy'), allow_pickle=True)

    y_pred = model.predict(X_test)

    with dvclive.Live(dvclive_dir, resume=True) as live:
        live.log_metric("accuracy", accuracy_score(y_test, y_pred))
        live.log_metric("precision_weighted", precision_score(y_test, y_pred, average='weighted', zero_division=0))
        live.log_metric("recall_weighted", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        live.log_metric("f1_weighted", f1_score(y_test, y_pred, average='weighted', zero_division=0))

        report = classification_report(y_test, y_pred, zero_division=0)
        live.log_plot("classification_report.txt", report)
        print("\nModel Performance Report:")
        print(report)

    print(f"Model testing completed. Metrics saved in {dvclive_dir}")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
    run_model_testing(base_path=project_dir)
