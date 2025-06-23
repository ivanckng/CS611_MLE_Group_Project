# after feature engineering and label engineering
# we need to process the data,like contact the feature data and label data
# and the data that splited by date should be store in somewhere like GCP bucket or local data folder
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


start_date_str = '2015-01-01'
end_date_str = '2017-03-01'
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        current_year = current_date.year
        current_month = current_date.month
        if current_month <10:
            current_date_str = f"{current_year}_0{current_month}_01"
        else:
            current_date_str = f"{current_year}_{current_month}_01"
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date_str)
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    

    return first_of_month_dates

dates_str = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str)


### Ingesting data from Google Cloud Storage

bucket_name = "cs611_mle"
label_path_in_bucket = "Gold Layer/labels.csv"
label_gcs_path = f"gs://{bucket_name}/{label_path_in_bucket}"

label_pdf = pd.read_csv(label_gcs_path)

label_pdf = label_pdf.drop_duplicates(subset = ['msno', 'membership_start_date', 'membership_expire_date'], keep='first', inplace=False)
label_pdf['membership_expire_date'] = pd.to_datetime(label_pdf['membership_expire_date'], format='%Y-%m-%d')
label_pdf['membership_start_date'] = pd.to_datetime(label_pdf['membership_start_date'], format='%Y-%m-%d')
label_pdf['plan_days'] = label_pdf['membership_expire_date'] -  label_pdf['membership_start_date']

feature_path_in_bucket = "datamart/gold/feature_store"
feature_gcs_path = [f"gs://{bucket_name}/{feature_path_in_bucket}/gold_featurestore_{date_str}.parquet/" for date_str in dates_str]

df_dict = {}
for path in feature_gcs_path:
    path_index = feature_gcs_path.index(path)
    date_str = dates_str[path_index]
    df_dict[f'df_{date_str}'] = pd.read_parquet(path)

feature_pdf = pd.concat(df_dict.values(), ignore_index=True)
feature_pdf['plan_days'] = feature_pdf['membership_expire_date'] -  feature_pdf['membership_start_date']

# target_plan_days = ['31 days', '30 days']
# feature_pdf = feature_pdf[feature_pdf['plan_days'].isin(target_plan_days)]
feature_pdf = feature_pdf[feature_pdf['plan_days'].isin([pd.Timedelta(days=30), pd.Timedelta(days=31)])]
feature_pdf = feature_pdf.drop_duplicates(subset = ['msno', 'membership_start_date', 'membership_expire_date'], keep='first', inplace=False)

label_pdf['membership_start_date'] = pd.to_datetime(label_pdf['membership_start_date'])
label_pdf['membership_expire_date'] = pd.to_datetime(label_pdf['membership_expire_date'])
feature_pdf['membership_start_date'] = pd.to_datetime(feature_pdf['membership_start_date'])
feature_pdf['membership_expire_date'] = pd.to_datetime(feature_pdf['membership_expire_date'])

final_pdf = pd.merge(label_pdf, feature_pdf, how = 'left', on = ['msno', 'membership_start_date', 'membership_expire_date'])
final_pdf = final_pdf.dropna()
final_pdf = final_pdf.reset_index()
final_pdf = final_pdf.drop(columns = ['index', 'plan_days_x', 'plan_days_y', 'transaction_date'])

final_pdf['registered_via'] = final_pdf['registered_via'].astype('category')
final_pdf['churn'] = final_pdf['churn'].astype('category')
final_pdf['payment_method_id'] = final_pdf['payment_method_id'].astype('category')
final_pdf['is_auto_renew'] = final_pdf['is_auto_renew'].astype('category')
final_pdf['is_cancel'] = final_pdf['is_cancel'].astype('category')
final_pdf['city'] = final_pdf['city'].astype('category')
final_pdf['gender'] = final_pdf['gender'].astype('category')

def upload_to_gcs(local_file_path, bucket_name, blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_file_path)
        print(f"Successfully uploaded {local_file_path} to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"Error uploading {local_file_path}: {e}")


GCS_BUCKET_NAME = 'cs611_mle'
GCS_BASE_PATH = 'datamart/gold'

base_dir = './datamart/gold'

oot_output_path = os.path.join(base_dir, 'OOT_data', 'oot_data.parquet')
train_output_path = os.path.join(base_dir, 'train_data', 'train_data.parquet')
inference_output_dir = os.path.join(base_dir, 'inference_data')
os.makedirs(os.path.dirname(oot_output_path), exist_ok=True)
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
os.makedirs(inference_output_dir, exist_ok=True)

### OOT
OOT_pdf = final_pdf[(final_pdf['membership_start_date'] >= '2016-11-01')&(final_pdf['membership_start_date'] <= '2016-11-30')]
OOT_pdf.to_parquet(oot_output_path,index=False)
# Ensure membership_expire_date is datetime
if not pd.api.types.is_datetime64_any_dtype(OOT_pdf['membership_expire_date']):
    OOT_pdf['membership_expire_date'] = pd.to_datetime(OOT_pdf['membership_expire_date'])

upload_to_gcs(oot_output_path, GCS_BUCKET_NAME, f'{GCS_BASE_PATH}/OOT_data/OOT_data.parquet')

### Train
model_pdf = final_pdf[(final_pdf['membership_start_date'] < '2016-11-01')]
model_pdf.to_parquet(train_output_path,index=False)
# Ensure membership_expire_date is datetime
if not pd.api.types.is_datetime64_any_dtype(model_pdf['membership_expire_date']):
    model_pdf['membership_expire_date'] = pd.to_datetime(model_pdf['membership_expire_date'])

upload_to_gcs(train_output_path, GCS_BUCKET_NAME, f'{GCS_BASE_PATH}/train_data/train_data.parquet')

### Inference
inference_pdf = final_pdf[(final_pdf['membership_start_date'] > '2016-12-01')]

# Ensure membership_expire_date is datetime
if not pd.api.types.is_datetime64_any_dtype(inference_pdf['membership_expire_date']):
    inference_pdf['membership_expire_date'] = pd.to_datetime(inference_pdf['membership_expire_date'])

# Split the inference data by membership_expire_date and store each subset
for expire_date, subset in inference_pdf.groupby(inference_pdf['membership_expire_date']):
    date_str = expire_date.strftime('%Y_%m_%d')
    file_path = os.path.join(inference_output_dir, f'inference_data_{date_str}.parquet')
    subset.to_parquet(file_path, index=False)
    print(f"Saved {file_path}, shape: {subset.shape}")
    gcs_blob_name = f'{GCS_BASE_PATH}/inference_data/inference_data_{date_str}.parquet'
    upload_to_gcs(file_path, GCS_BUCKET_NAME, gcs_blob_name)