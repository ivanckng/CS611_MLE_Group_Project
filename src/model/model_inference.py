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

# Path to the pre-split daily inference dataset (2017-01-01)
inference_daily_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "datamart",
    "gold",
    "inference_data",
    "inference_data_2017_01_01.parquet",
)

# Normalise the path (.. components might remain from the join above)
inference_daily_path = os.path.normpath(inference_daily_path)

if not os.path.exists(inference_daily_path):
    raise FileNotFoundError(f"Daily inference file not found: {inference_daily_path}")
inference_pdf = pd.read_parquet(inference_daily_path)

# inference_pdf=pd.read_parquet('../../datamart/gold/inference_data/inference_data.parquet')

inference_pdf = inference_pdf.drop(columns = ['msno', 'membership_start_date', 'membership_expire_date', 'churn'])

cate_cols = ['payment_method_id', 'is_auto_renew', 'is_cancel', 'city', 'gender', 'registered_via']
inference_numeric = inference_pdf.drop(columns = cate_cols)
inference_cate = inference_pdf[cate_cols]

scaler = StandardScaler()
transformer_stdscaler = scaler.fit(inference_numeric)
inference_num_processed = pd.DataFrame(transformer_stdscaler.transform(inference_numeric), columns=inference_numeric.columns, index = inference_numeric.index)

methods = ['41', '40', '36', '39', '37']

def method_mapping(col):
    if col in methods:
        return col
    else:
        return '99'
     

inference_cate['payment_method_id'] = inference_cate['payment_method_id'].apply(method_mapping)

cities = ['1', '13', '5', '4', '15', '22']

def city_mapping(col):
    if col in cities:
        return col
    else:
        return '99'
     

inference_cate['city'] = inference_cate['city'].apply(city_mapping)

# deal with registered_via
regist = ['7', '9', '3']

def regist_mapping(col):
    if col in regist:
        return col
    else:
        return '99'
     

inference_cate['registered_via'] = inference_cate['registered_via'].apply(regist_mapping)
inference_cate['payment_method_id'] = inference_cate['payment_method_id'].astype('category')
inference_cate['is_auto_renew'] = inference_cate['is_auto_renew'].astype('category')
inference_cate['is_cancel'] = inference_cate['is_cancel'].astype('category')
inference_cate['city'] = inference_cate['city'].astype('category')
inference_cate['gender'] = inference_cate['gender'].astype('category')

inference_cate_processed = pd.get_dummies(inference_cate, columns=['payment_method_id', 'city', 'gender', 'registered_via'],dtype=int)
inference_features = pd.concat([inference_num_processed, inference_cate_processed], axis=1)

inference_features = pd.get_dummies(inference_features, drop_first=True)
print(inference_features.info())

#----------------------------------------------------------
#batch prediction for a single-day inference dataset
#----------------------------------------------------------

url = 'http://0.0.0.0:8000/predict_batch/'
# display(inference_features.head())
predictions = []
# print(inference_features.iloc[0].to_string())
for index,record in inference_features.iterrows():
    record=record.to_dict()
    payload = {'features': record}
    payload = json.dumps(payload)
    response = requests.post(url, data=payload)
    predictions.append(response.json())
    print(response.json())

print(predictions)

#----------------------------------------------------------
#batch prediction for a single-day inference dataset
#----------------------------------------------------------

batch_payload = {"instances": inference_features.to_dict(orient="records")}
batch_endpoint = "http://0.0.0.0:8000/predict_batch"

try:
    batch_response = requests.post(batch_endpoint, json=batch_payload, timeout=60)
    batch_response.raise_for_status()
    print("\nBatch prediction for 2017-01-01 completed successfully – first 10 results:")
    print(batch_response.json()["predictions"])
except requests.RequestException as req_err:
    print(f"Batch prediction request failed: {req_err}")
    if req_err.response is not None:
        print("Server response:", req_err.response.text)