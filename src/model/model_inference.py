#put inference code here, the current inference code is in client.py and client.ipynb
'''
Script for inferencing the deployed model
'''

import json
import requests

import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform, loguniform

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from google.cloud import storage

import argparse
import tempfile

bucket_name = "cs611_mle"
# CONSTANTS ------------------------------------------------------------------
try:
    import google.colab
    from google.colab import auth
    auth.authenticate_user()
    project_id = "sound-memory-457612-b8"
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
except:
    pass
DEFAULT_INPUT_PREFIX = os.getenv(
    "GCS_INFERENCE_PREFIX", "datamart/gold/inference_data"
)
DEFAULT_OUTPUT_PREFIX = os.getenv(
    "GCS_PREDICTION_PREFIX", "datamart/gold/model_predictions"
)
PRED_ENDPOINT = os.getenv("PREDICTION_ENDPOINT", "http://0.0.0.0:8000/predict_batch")



# CORE LOGIC -----------------------------------------------------------------

def run_inference(date_str: str) -> None:
    """Main entry for running inference for a given date (YYYY-MM-DD)."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    formatted_date = date_obj.strftime("%Y_%m_%d")

    # Build GCS paths
    input_blob_path = f"gs://{bucket_name}/{DEFAULT_INPUT_PREFIX}/inference_data_{formatted_date}.parquet"
    output_blob_path = f"gs://{bucket_name}/{DEFAULT_OUTPUT_PREFIX}/prediction_{formatted_date}.parquet"


    inference_pdf = pd.read_parquet(input_blob_path)
    if "churn" not in inference_pdf.columns:
        raise ValueError("Expected 'churn' column to be present in inference data.")

    daily_target = inference_pdf["churn"].copy()
    inference_pdf = inference_pdf.drop(
        columns=["msno", "membership_start_date", "membership_expire_date", "churn"],
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # Pre-processing (mirrors training pipeline)
    # ------------------------------------------------------------------
    cate_cols = [
        "payment_method_id",
        "is_auto_renew",
        "is_cancel",
        "city",
        "gender",
        "registered_via",
    ]
    inference_numeric = inference_pdf.drop(columns=cate_cols)
    inference_cate = inference_pdf[cate_cols]

    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(inference_numeric)
    inference_num_processed = pd.DataFrame(
        transformer_stdscaler.transform(inference_numeric),
        columns=inference_numeric.columns,
        index=inference_numeric.index,
    )

    # Categorical value grouping --------------------------------------
    methods = ["41", "40", "36", "39", "37"]
    inference_cate["payment_method_id"] = inference_cate["payment_method_id"].apply(
        lambda x: x if x in methods else "99"
    )

    cities = ["1", "13", "5", "4", "15", "22"]
    inference_cate["city"] = inference_cate["city"].apply(
        lambda x: x if x in cities else "99"
    )

    regist = ["7", "9", "3"]
    inference_cate["registered_via"] = inference_cate["registered_via"].apply(
        lambda x: x if x in regist else "99"
    )

    # Cast to categorical dtypes
    cat_dtype_cols = [
        "payment_method_id",
        "is_auto_renew",
        "is_cancel",
        "city",
        "gender",
    ]
    for col in cat_dtype_cols:
        inference_cate[col] = inference_cate[col].astype("category")

    # One-hot encode categorical columns
    inference_cate_processed = pd.get_dummies(
        inference_cate,
        columns=["payment_method_id", "city", "gender", "registered_via"],
        dtype=int,
    )

    # Combine features
    inference_features = pd.concat(
        [inference_num_processed, inference_cate_processed], axis=1
    )
    inference_features = pd.get_dummies(inference_features, drop_first=True)

    print(
        f"Prepared feature matrix with shape {inference_features.shape} – sending to prediction endpoint..."
    )

    # ------------------------------------------------------------------
    # Batch prediction request ----------------------------------------
    # ------------------------------------------------------------------
    batch_payload = {"instances": inference_features.to_dict(orient="records")}
    try:
        response = requests.post(PRED_ENDPOINT, json=batch_payload, timeout=120)
        response.raise_for_status()
        preds = response.json().get("predictions")
        if preds is None:
            raise ValueError("No 'predictions' key returned from endpoint response.")
    except requests.RequestException as exc:
        raise RuntimeError(f"Prediction request failed: {exc}")

    if len(preds) != len(daily_target):
        raise ValueError(
            "Mismatch between number of predictions and target rows – cannot continue."
        )

    # ------------------------------------------------------------------
    # Persist results back to GCS --------------------------------------
    # ------------------------------------------------------------------
    result_df = pd.DataFrame({"prediction": preds, "churn": daily_target.values})

    print(
        f"Uploading prediction parquet to gs://{bucket_name}/{output_blob_path} (rows: {len(result_df)})..."
    )
    result_df.to_parquet(output_blob_path)
    # upload_file_to_gcs(bucket, output_blob_path)


# ENTRYPOINT -----------------------------------------------------------------
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