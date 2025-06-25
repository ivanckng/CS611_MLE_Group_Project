import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import gcsfs

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


### Load Gold Table

fs = gcsfs.GCSFileSystem()
gcs_path = 'gs://cs611_mle/datamart/gold/train_data/train_data.parquet'
df = pd.read_parquet(gcs_path, filesystem=fs)

### Split Data

y_model_ps = df['churn']
x_model_pdf = df.drop(columns = ['msno', 'membership_start_date', 'membership_expire_date', 'churn'])

x_train, x_test,y_train, y_test = train_test_split(x_model_pdf, y_model_ps, test_size= 0.2, shuffle=True, random_state=611, stratify=y_model_ps)

### Preprocessing 
cate_cols = ['payment_method_id', 'is_auto_renew', 'is_cancel', 'city', 'gender', 'registered_via']
x_train_numeric = x_train.drop(columns = cate_cols)
x_test_numeric = x_test.drop(columns = cate_cols)

x_train_cate = x_train[cate_cols]
x_test_cate = x_test[cate_cols]

# Standardisation for Numeric Columns
scaler = StandardScaler()

transformer_stdscaler = scaler.fit(x_train_numeric)

x_train_num_processed = pd.DataFrame(transformer_stdscaler.transform(x_train_numeric), columns=x_train_numeric.columns, index = x_train_numeric.index)
x_test_num_processed = pd.DataFrame(transformer_stdscaler.transform(x_test_numeric), columns=x_test_numeric.columns, index = x_test_numeric.index)

# Categorical Data
# check payment_method_id
x_train_cate['payment_method_id'].value_counts(normalize=True).cumsum() * 100
# deal with city
methods = ['41', '40', '36', '39', '37']

def method_mapping(col):
    if col in methods:
        return col
    else:
        return '99'
     

x_train_cate['payment_method_id'] = x_train_cate['payment_method_id'].apply(method_mapping)
x_test_cate['payment_method_id'] = x_test_cate['payment_method_id'].apply(method_mapping)

# Categorical Data
# check cities
x_train_cate['city'].value_counts(normalize=True).cumsum() * 100

# deal with city
cities = ['1', '13', '5', '4', '15', '22']

def city_mapping(col):
    if col in cities:
        return col
    else:
        return '99'
     

x_train_cate['city'] = x_train_cate['city'].apply(city_mapping)
x_test_cate['city'] = x_test_cate['city'].apply(city_mapping)

# Categorical Data
# check registered_via
x_train_cate['registered_via'].value_counts(normalize=True).cumsum() * 100
# deal with registered_via
regist = ['7', '9', '3']

def regist_mapping(col):
    if col in regist:
        return col
    else:
        return '99'
     

x_train_cate['registered_via'] = x_train_cate['registered_via'].apply(regist_mapping)
x_test_cate['registered_via'] = x_test_cate['registered_via'].apply(regist_mapping)

x_train_cate_processed = pd.get_dummies(x_train_cate, columns=['payment_method_id', 'city', 'gender', 'registered_via'], dtype=int)
x_test_cate_processed = pd.get_dummies(x_test_cate, columns=['payment_method_id', 'city', 'gender', 'registered_via'], dtype=int)

x_train = pd.concat([x_train_num_processed, x_train_cate_processed], axis=1)
x_test = pd.concat([x_test_num_processed, x_test_cate_processed], axis=1)

x_train = pd.get_dummies(x_train, drop_first=True)
x_test = pd.get_dummies(x_test, drop_first=True)

### Hyperparameters Tuning
## RF
try:
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    
    # only show warning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def rf_objective(trial):
        # params space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.8]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
        
        # create model
        rf_model = RandomForestClassifier(
            random_state=42, 
            n_jobs=-1,
            **params
        )
        
        # Cross validation
        scores = cross_val_score(
            rf_model, x_train, y_train, 
            cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        return scores.mean()
    
    study = optuna.create_study(
        direction='maximize',  # maximize AUC
        sampler=TPESampler(
            seed=42,
            n_startup_trials=2, 
            n_ei_candidates=24 
        )
    )
    
    # 5 trails
    print("Starting Bayesian Optimization with Optuna (5 trials)...")
    study.optimize(rf_objective, n_trials=5, show_progress_bar=True)
    
    # best params
    rf_best_params_optuna = study.best_params
    rf_best_score_optuna = study.best_value
    
    # Train the model with the best parameters and evaluate it on the validation set
    rf_best_model_optuna = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1,
        **rf_best_params_optuna
    )
    rf_best_model_optuna.fit(x_train, y_train)
    rf_test_pred_optuna = rf_best_model_optuna.predict_proba(x_test)[:, 1]
    rf_test_auc_optuna = roc_auc_score(y_test, rf_test_pred_optuna)
    
    # output
    print(f"\nRF Optuna Optimization Results (5 trials):")
    print(f"  Best CV Score: {rf_best_score_optuna:.4f}")
    print(f"  Test AUC: {rf_test_auc_optuna:.4f}")
    print(f"  Best Parameters: {rf_best_params_optuna}")
    
    # results of 5 trails
    print(f"\nAll trial results:")
    for i, trial in enumerate(study.trials):
        print(f"  Trial {i+1}: AUC = {trial.value:.4f}, Params = {trial.params}")
    
except Exception as e:
    print(f"Error in RF Optuna Optimization: {e}")
    rf_test_auc_optuna = 0


## LR

# Convert y to the format suitable for Logistic Regression# import numpy as np

if hasattr(y_train, 'cat'):
    y_train_lr = y_train.cat.codes.values
else:
    y_train_lr = y_train

y_train_lr = np.where(y_train_lr == -1, 0, y_train_lr)

y_train_lr = y_train_lr.astype(int)

#Bayesian optimization for LR using Optuna TPE

try:
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    
    # only show warning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def lr_objective(trial):
        try:
            # params space
            params = {
                'C': trial.suggest_float('C', 0.1, 50, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
            }
            
            # create LR model
            lr_model = LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                solver='liblinear',  # liblinear surpport l1 & l2 
                **params
            )
            
            # cross valisation
            scores = cross_val_score(
                lr_model, x_train, y_train_lr, 
                cv=3, scoring='roc_auc', n_jobs=-1
            )
            
            return scores.mean() # mean of AUC
            
        except Exception as e:
            print("Failed params:", params)
            print("Error:", e)
            return 0.0 
    
    study = optuna.create_study(
        direction='maximize', 
        sampler=TPESampler(
            seed=42,
            n_startup_trials=3,   
            n_ei_candidates=24   
        )
    )
    
    print("Starting LR Bayesian Optimization with Optuna...")
    study.optimize(lr_objective, n_trials=5, show_progress_bar=True)
    
    # best params
    lr_best_params_bo = study.best_params
    lr_best_score_bo = study.best_value
    
    # Train the model with the best parameters and evaluate it on the validation set
    lr_best_model_bo = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear',
        **lr_best_params_bo
    )
    
    lr_best_model_bo.fit(x_train, y_train_lr)
    lr_test_pred_bo = lr_best_model_bo.predict_proba(x_test)[:, 1]
    lr_test_auc_bo = roc_auc_score(y_test, lr_test_pred_bo)
    
    # results
    print(f"\nLR Optuna Optimization Results:")
    print(f"  Best CV Score: {lr_best_score_bo:.4f}")
    print(f"  Test AUC: {lr_test_auc_bo:.4f}")
    print(f"  Best Parameters: {lr_best_params_bo}")
    
    # results for 5 trials
    print(f"\nAll trial results:")
    for i, trial in enumerate(study.trials):
        print(f"  Trial {i+1}: AUC = {trial.value:.4f}, Params = {trial.params}")
    
    bayesian_available = True
    
except ImportError:
    print("Optuna not available. Please install: pip install optuna")
    bayesian_available = False
    lr_test_auc_bo = 0  
except Exception as e:
    print(f"Error in LR Optuna Optimization: {e}")
    bayesian_available = False
    lr_test_auc_bo = 0


### Find best threshold (based on Fβ(1.5))

#RF
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# threshold space
thresholds = np.arange(0.1, 0.9, 0.01)
results = []

# Train set prediction probability for RF
rf_train_pred = rf_best_model_optuna.predict_proba(x_train)[:, 1]

# Test set prediction probability for RF
rf_test_pred = rf_best_model_optuna.predict_proba(x_test)[:, 1]

# find the best threshold based on F1.5
for thresh in thresholds:
   y_pred = (rf_train_pred >= thresh).astype(int)
   f1_5 = fbeta_score(y_train, y_pred, beta=1.5)
   prec = precision_score(y_train, y_pred)
   rec = recall_score(y_train, y_pred)
   tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
   
   # Calculate FN/(FN+TN) - False Omission Rate
   fn_rate = fn / (fn + tn) if (fn + tn) > 0 else 0
   
   results.append((thresh, f1_5, fn_rate)) 

df_thresh = pd.DataFrame(results, columns=[
   "Threshold", "F1.5", "FN_Rate"
])

# Find best threshold based on F1.5
best_threshold = df_thresh.loc[df_thresh["F1.5"].idxmax(), "Threshold"] 
print("Best threshold metrics:")
print(df_thresh.loc[df_thresh["F1.5"].idxmax()])
rf_threshold = best_threshold

# Evaluate on the test set using this optimal threshold
y_test_pred = (rf_test_pred >= best_threshold).astype(int)

# Calculate final metrics: F1.5, AUC, and FN/(FN+TN)
test_f1_5 = fbeta_score(y_test, y_test_pred, beta=1.5)
test_auc = roc_auc_score(y_test, rf_test_pred)

# Calculate FN/(FN+TN) for test set
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
test_fn_rate = fn_test / (fn_test + tn_test) if (fn_test + tn_test) > 0 else 0

print(f"\n=== Final Test Set Metrics ===")
print(f"Optimal threshold: {best_threshold:.2f}")
print(f"F1.5-score: {test_f1_5:.4f}")
print(f"AUC: {test_auc:.4f}")
print(f"FN/(FN+TN): {test_fn_rate:.4f}")


#LR
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# threshold space
thresholds = np.arange(0.1, 0.9, 0.01)
results = []

# Train set prediction probability for LR
lr_train_pred = lr_best_model_bo.predict_proba(x_train)[:, 1]

# Test set prediction probability for LR
lr_test_pred = lr_best_model_bo.predict_proba(x_test)[:, 1]

# find the best threshold based on F1.5
for thresh in thresholds:
   preds = (lr_train_pred >= thresh).astype(int)
   f1_5 = fbeta_score(y_train_lr, preds, beta=1.5)  
   prec = precision_score(y_train_lr, preds)
   rec = recall_score(y_train_lr, preds)
   tn, fp, fn, tp = confusion_matrix(y_train_lr, preds).ravel()
   
   # Calculate FN/(FN+TN) - False Omission Rate
   fn_rate = fn / (fn + tn) if (fn + tn) > 0 else 0
   
   results.append((thresh, f1_5, fn_rate))

df_thresh_lr = pd.DataFrame(results, columns=[
   "Threshold", "F1.5", "FN_Rate"
])

# Find best threshold based on F1.5 (instead of Recall)
best_threshold = df_thresh_lr.loc[df_thresh_lr["F1.5"].idxmax(), "Threshold"]
print("Best threshold metrics:")
print(df_thresh_lr.loc[df_thresh_lr["F1.5"].idxmax()])
lr_threshold = best_threshold

# Evaluate on the test set using this optimal threshold
y_test_pred = (lr_test_pred >= lr_threshold).astype(int)

# Calculate final metrics: F1.5, AUC, and FN/(FN+TN)
test_f1_5 = fbeta_score(y_test, y_test_pred, beta=1.5)
test_auc = roc_auc_score(y_test, lr_test_pred)

# Calculate FN/(FN+TN) for test set
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
test_fn_rate = fn_test / (fn_test + tn_test) if (fn_test + tn_test) > 0 else 0

print(f"\n=== Final Test Set Metrics ===")
print(f"Optimal threshold: {best_threshold:.2f}")
print(f"F1.5-score: {test_f1_5:.4f}")
print(f"AUC: {test_auc:.4f}")
print(f"FN/(FN+TN): {test_fn_rate:.4f}")


import joblib

# save model path
model_dir = "models"
if not os.path.exists(model_dir):
   os.makedirs(model_dir)

# save LR model, threshold, and scaler
if 'lr_best_model_bo' in locals() and lr_best_model_bo is not None:
   
   # Save LR model (joblib)
   lr_joblib_path = f"{model_dir}/lr_best_model_.joblib"
   joblib.dump(lr_best_model_bo, lr_joblib_path)
   
   # Save optimal threshold (joblib)
   lr_threshold_path = f"{model_dir}/lr_threshold.joblib"
   joblib.dump(lr_threshold, lr_threshold_path)
   
   # Save scaler (joblib)
   if 'scaler' in locals() and scaler is not None:
       lr_scaler_path = f"{model_dir}/lr_scaler.joblib"
       joblib.dump(scaler, lr_scaler_path)
   else:
       lr_scaler_path = "Scaler not found"
       print("Warning: Scaler not found in locals()")
   
   # Save model info (joblib)
   lr_info = {
       'model_type': 'LogisticRegression',
       'best_params': lr_best_params_bo,
       'cv_score': lr_best_score_bo,
       'test_auc': test_auc, 
       'test_f1_5': test_f1_5, 
       'test_fn_rate': test_fn_rate, 
       'threshold': lr_threshold,
       'sample_size': len(x_train),
       'feature_count': x_train.shape[1] if hasattr(x_train, 'shape') else 'unknown',
       'feature_names': list(x_train.columns) if hasattr(x_train, 'columns') else 'unknown'
   }
   
   lr_info_path = f"{model_dir}/lr_model_info.joblib"
   joblib.dump(lr_info, lr_info_path)
   
   print(f"LogisticRegression model and components saved")
   
else:
   print("LogisticRegression model not found or optimization failed")
print()

   print(f"RandomForest model and components saved")

else:
   print("RandomForest model not found or optimization failed")
