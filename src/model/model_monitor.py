import argparse
from datetime import datetime
import pandas as pd
from google.cloud import storage
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score

def run_inference(date_str):
    # Identify the date
    try:
        formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y_%m_%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

    # GCS path
    bucket_name = "cs611_mle"
    gcs_path = f"gs://{bucket_name}/datamart_old/gold/model_predictions/prediction_{formatted_date}.parquet"

    df = pd.read_parquet(gcs_path)

    print(f"[INFO] Loaded {len(df)} rows from prediction data for {date_str}")
    
    # y_pred
    y_pred = df.prediction
    # y_true
    y_true = df.churn

    ### Evaluation
    def calculate_fn_rate(y_true, y_pred):
    
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
        # False Omission Rate: FN/(TN+FN)
        if (tn + fn) > 0:
            fn_rate = fn / (tn + fn)
        else:
            fn_rate = 0 
    
        return fn_rate

    fn_rate = calculate_fn_rate(y_true, y_pred)
    f1_5 = fbeta_score(y_true, y_pred, beta=1.5)
    auc = roc_auc_score(y_true, y_pred)

    print(f"[INFO] FN Rate: {fn_rate:.4f}")
    print(f"[INFO] F1.5 Score: {f1_5:.4f}")
    print(f"[INFO] AUC Score: {auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run daily churn inference and persist results to GCS."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date of the inference dataset to process in YYYY-MM-DD format.",
    )
    args = parser.parse_args()

    run_inference(args.date)
