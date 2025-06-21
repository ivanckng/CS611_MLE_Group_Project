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


inference_pdf = final_pdf[(final_pdf['membership_start_date'] > '2016-12-01')]
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
inference_cate_processed = pd.get_dummies(inference_cate, columns=['payment_method_id', 'city', 'gender', 'registered_via'], dtype=int)

inference_features = pd.concat([inference_num_processed, inference_cate_processed], axis=1)

url = 'http://0.0.0.0:8000/predict/'


predictions = []
for record in inference_pdf[:2]:
    payload = {'features': record.to_dict()}
    payload = json.dumps(payload)
    response = requests.post(url, data=payload)
    predictions.append(response.json())

print(predictions)


