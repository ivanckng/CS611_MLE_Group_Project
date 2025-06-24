import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
from google.cloud import storage 

def model_inference(**context):
    date_str = context["ds"]
    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y_%m_%d")

    # GCS path
    bucket_name = "cs611_mle"
    path_in_bucket = f"gs://{bucket_name}/datamart/gold/model_predictions/prediction_{formatted_date}.parquet"

    # Read the inference prediction
    df = pd.read_parquet(path_in_bucket)

    # Evaluation
    y_pred = df["prediction"]
    y_true = df["churn"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if (tn + fn) > 0:
        fn_rate = fn / (tn + fn)
    else:
        fn_rate = 0

    print(f"[METRIC] False Omission Rate (FN/(TN+FN)): {fn_rate:.4f}")
