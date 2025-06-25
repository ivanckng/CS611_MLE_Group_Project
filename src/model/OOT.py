import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import gcsfs
import joblib
from sklearn.metrics import roc_curve, auc, fbeta_score, precision_score, recall_score, confusion_matrix



### Load OOT set

fs = gcsfs.GCSFileSystem()
gcs_path = 'gs://cs611_mle/datamart/gold/OOT_data/OOT_data.parquet'
df = pd.read_parquet(gcs_path, filesystem=fs)

y_true = df['churn']
# preprocessing
exclude_cols = ['msno', 'membership_start_date', 'membership_expire_date', 'churn']
x = df.drop(columns=exclude_cols)
cate_cols = ['payment_method_id', 'is_auto_renew', 'is_cancel', 'city', 'gender', 'registered_via']
x_n = x.drop(columns=cate_cols)
x_c = x[cate_cols].copy()

# standardization
scaler_path = "models/lr_scaler.joblib"
try:
    scaler = joblib.load(scaler_path)
    print("Scaler Load succesfully")
except Exception as e:
    print(f"Scaler load faild: {e}")
    scaler = None
x_n = pd.DataFrame(scaler.transform(x_n), columns=x_n.columns, index=x_n.index)

x_c['payment_method_id'].value_counts(normalize=True).cumsum() * 100
# deal with city
methods = ['41', '40', '36', '39', '37']

def method_mapping(col):
    if col in methods:
        return col
    else:
        return '99'
     

x_c['payment_method_id'] = x_c['payment_method_id'].apply(method_mapping)


x_c['city'].value_counts(normalize=True).cumsum() * 100
# deal with city
cities = ['1', '13', '5', '4', '15', '22']

def city_mapping(col):
    if col in cities:
        return col
    else:
        return '99'
     

x_c['city'] = x_c['city'].apply(city_mapping)

x_c['registered_via'].value_counts(normalize=True).cumsum() * 100
# deal with registered_via
regist = ['7', '9', '3']

def regist_mapping(col):
    if col in regist:
        return col
    else:
        return '99'
     

x_c['registered_via'] = x_c['registered_via'].apply(regist_mapping)

x_infer_cate_processed = pd.get_dummies(x_c, columns=['payment_method_id', 'city', 'gender', 'registered_via'], dtype=int)
x_infer = pd.concat([x_n, x_infer_cate_processed], axis=1)
x = x_infer


try:
    lr_best_model = joblib.load('saved_models/lr_best_model_.joblib')
    print("Model load succesfully！")
except FileNotFoundError:
    print("Not Find")
except Exception as e:
    print(f"Error: {e}")


proba = lr_best_model.predict_proba(x) 
lr_threshold = joblib.load("saved_models/lr_threshold.joblib")
y_pred = (proba[:, 1] >= lr_threshold).astype(int)

#Evaluation
def calculate_fn_rate(y_true, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # False Omission Rate: FN/(TN+FN)
    if (tn + fn) > 0:
        fn_rate = fn / (tn + fn)
    else:
        fn_rate = 0 
    
    return fn_rate

fn_rate = calculate_fn_rate(y_true, y_pred)

print('Fn_Rate: FN/(TN+FN)',fn_rate)

