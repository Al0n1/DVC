import numpy as np
import joblib
import os
import dvclive
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_model_testing(base_path: str):
    models_dir = os.path.join(base_path, 'models')
    data_processed_dir = os.path.join(base_path, 'data', 'processed')

    model_path = os.path.join(models_dir, 'rf_model.joblib')
    model = joblib.load(model_path)

    X_test = np.load(os.path.join(data_processed_dir, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_processed_dir, 'y_test.npy'), allow_pickle=True)

    y_pred = model.predict(X_test)

    with dvclive.Live(resume=True) as live:
        live.log_metric("accuracy", accuracy_score(y_test, y_pred))
        live.log_metric("precision_weighted", precision_score(y_test, y_pred, average='weighted', zero_division=0))
        live.log_metric("recall_weighted", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        live.log_metric("f1_weighted", f1_score(y_test, y_pred, average='weighted', zero_division=0))

        report_str = classification_report(y_test, y_pred, zero_division=0)
        print("\nModel Performance Report:")
        print(report_str)

        artifacts_log_dir = os.path.join(live.dir, "artifacts") 
        os.makedirs(artifacts_log_dir, exist_ok=True)
        report_file_path = os.path.join(artifacts_log_dir, "classification_report.txt")

        with open(report_file_path, "w") as f:
            f.write(report_str)

        live.log_artifact(report_file_path, name="classification-report")

    print(f"Model testing completed. Metrics and artifacts saved in {live.dir}")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
    run_model_testing(base_path=project_dir)
