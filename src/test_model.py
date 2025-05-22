import numpy as np
import joblib
import os
import dvclive
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_model_testing(base_path: str):
    models_dir = os.path.join(base_path, 'models')
    data_processed_dir = os.path.join(base_path, 'data', 'processed')
    dvclive_dir = os.path.join(base_path, 'dvclive') # Ensure dvclive outputs to project root's dvclive folder

    os.makedirs(dvclive_dir, exist_ok=True)

    # Load model and data
    model_path = os.path.join(models_dir, 'rf_model.joblib')
    model = joblib.load(model_path)

    X_test = np.load(os.path.join(data_processed_dir, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_processed_dir, 'y_test.npy'), allow_pickle=True)

    # Make predictions
    y_pred = model.predict(X_test)

    # Initialize dvclive
    # dvclive.init() will create 'dvclive' directory if it doesn't exist
    # and will save metrics.json and plots inside it.
    # We pass the path to ensure it's in the project root.
    with dvclive.Live(dvclive_dir, resume=True) as live:
        # Log metrics
        live.log_metric("accuracy", accuracy_score(y_test, y_pred))
        live.log_metric("precision_weighted", precision_score(y_test, y_pred, average='weighted', zero_division=0))
        live.log_metric("recall_weighted", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        live.log_metric("f1_weighted", f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # Log classification report as a text artifact (optional, but can be useful)
        report = classification_report(y_test, y_pred, zero_division=0)
        live.log_text("classification_report.txt", report)
        print("\nModel Performance Report:")
        print(report)

        # Log confusion matrix
        # live.log_sklearn_plot("confusion_matrix", y_test, y_pred, name="confusion_matrix.png")
        # For newer dvclive versions, log_image is preferred for custom plots or plots not directly supported
        # For simplicity, we'll rely on the text report and individual metrics for now.
        # If a visual confusion matrix is strictly needed and log_sklearn_plot causes issues,
        # one might need to generate it with matplotlib and save it with live.log_image.

    print(f"Model testing completed. Metrics saved in {dvclive_dir}")

if __name__ == "__main__":
    # This block allows the script to be run standalone.
    # When DVC executes `python src/test_model.py`,
    # this __main__ block will correctly set `base_path` to the project root.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
    run_model_testing(base_path=project_dir)
