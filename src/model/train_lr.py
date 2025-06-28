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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform, loguniform

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import make_classification
import joblib

### Load Gold Table
print("Loading data from GCS...")
fs = gcsfs.GCSFileSystem()
gcs_path = 'gs://cs611_mle/datamart/gold/train_data/train_data.parquet'
df = pd.read_parquet(gcs_path, filesystem=fs)
print(f"Data loaded successfully. Shape: {df.shape}")

### Split Data
print("Splitting data...")
y_model_ps = df['churn']
x_model_pdf = df.drop(columns=['msno', 'membership_start_date', 'membership_expire_date', 'churn'])

x_train, x_test, y_train, y_test = train_test_split(
    x_model_pdf, y_model_ps, 
    test_size=0.2, 
    shuffle=True, 
    random_state=611, 
    stratify=y_model_ps
)

### Preprocessing 
print("Preprocessing data...")
cate_cols = ['payment_method_id', 'is_auto_renew', 'is_cancel', 'city', 'gender', 'registered_via']
x_train_numeric = x_train.drop(columns=cate_cols)
x_test_numeric = x_test.drop(columns=cate_cols)

x_train_cate = x_train[cate_cols]
x_test_cate = x_test[cate_cols]

# Standardisation for Numeric Columns
scaler = StandardScaler()
transformer_stdscaler = scaler.fit(x_train_numeric)

x_train_num_processed = pd.DataFrame(
    transformer_stdscaler.transform(x_train_numeric), 
    columns=x_train_numeric.columns, 
    index=x_train_numeric.index
)
x_test_num_processed = pd.DataFrame(
    transformer_stdscaler.transform(x_test_numeric), 
    columns=x_test_numeric.columns, 
    index=x_test_numeric.index
)

# Categorical Data Processing
# Payment method mapping
x_train_cate['payment_method_id'].value_counts(normalize=True).cumsum() * 100
methods = ['41', '40', '36', '39', '37']

def method_mapping(col):
    if col in methods:
        return col
    else:
        return '99'

x_train_cate['payment_method_id'] = x_train_cate['payment_method_id'].apply(method_mapping)
x_test_cate['payment_method_id'] = x_test_cate['payment_method_id'].apply(method_mapping)

# City mapping
x_train_cate['city'].value_counts(normalize=True).cumsum() * 100
cities = ['1', '13', '5', '4', '15', '22']

def city_mapping(col):
    if col in cities:
        return col
    else:
        return '99'

x_train_cate['city'] = x_train_cate['city'].apply(city_mapping)
x_test_cate['city'] = x_test_cate['city'].apply(city_mapping)

# Registered via mapping
x_train_cate['registered_via'].value_counts(normalize=True).cumsum() * 100
regist = ['7', '9', '3']

def regist_mapping(col):
    if col in regist:
        return col
    else:
        return '99'

x_train_cate['registered_via'] = x_train_cate['registered_via'].apply(regist_mapping)
x_test_cate['registered_via'] = x_test_cate['registered_via'].apply(regist_mapping)

# One-hot encoding
x_train_cate_processed = pd.get_dummies(
    x_train_cate, 
    columns=['payment_method_id', 'city', 'gender', 'registered_via'], 
    dtype=int
)
x_test_cate_processed = pd.get_dummies(
    x_test_cate, 
    columns=['payment_method_id', 'city', 'gender', 'registered_via'], 
    dtype=int
)

# Combine processed features
x_train = pd.concat([x_train_num_processed, x_train_cate_processed], axis=1)
x_test = pd.concat([x_test_num_processed, x_test_cate_processed], axis=1)

x_train = pd.get_dummies(x_train, drop_first=True)
x_test = pd.get_dummies(x_test, drop_first=True)

print(f"Final training data shape: {x_train.shape}")
print(f"Final test data shape: {x_test.shape}")

### Prepare target variable for Logistic Regression
# Convert y to the format suitable for Logistic Regression
if hasattr(y_train, 'cat'):
    y_train_lr = y_train.cat.codes.values
else:
    y_train_lr = y_train

y_train_lr = np.where(y_train_lr == -1, 0, y_train_lr)
y_train_lr = y_train_lr.astype(int)

### Enhanced Hyperparameter Tuning for Logistic Regression
try:
    import optuna
    from optuna.samplers import TPESampler
    
    # Only show warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def enhanced_lr_objective(trial):
        try:
            # Parameter space
            params = {
                'C': trial.suggest_float('C', 0.1, 50, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
            }
            
            # Create LR model
            lr_model = LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                solver='liblinear',  # liblinear supports l1 & l2 
                **params
            )
            
            # Manual cross-validation to get both train and val metrics
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            train_aucs = []
            val_aucs = []
            train_f1_5s = []
            val_f1_5s = []
            train_fn_rates = []
            val_fn_rates = []
            
            for train_idx, val_idx in cv.split(x_train, y_train_lr):
                X_train_fold = x_train.iloc[train_idx]
                X_val_fold = x_train.iloc[val_idx]
                y_train_fold = y_train_lr[train_idx]
                y_val_fold = y_train_lr[val_idx]
                
                # Fit model
                lr_model.fit(X_train_fold, y_train_fold)
                
                # Predictions
                train_preds = lr_model.predict_proba(X_train_fold)[:, 1]
                val_preds = lr_model.predict_proba(X_val_fold)[:, 1]
                
                # AUC scores
                train_auc = roc_auc_score(y_train_fold, train_preds)
                val_auc = roc_auc_score(y_val_fold, val_preds)
                
                # F1.5 scores and FN rate (using 0.5 as default threshold)
                train_preds_binary = (train_preds >= 0.5).astype(int)
                val_preds_binary = (val_preds >= 0.5).astype(int)
                
                train_f1_5 = fbeta_score(y_train_fold, train_preds_binary, beta=1.5)
                val_f1_5 = fbeta_score(y_val_fold, val_preds_binary, beta=1.5)
                
                # Calculate FN rate = FN / (TN + FN)
                tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_fold, train_preds_binary).ravel()
                tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_fold, val_preds_binary).ravel()
                
                train_fn_rate = fn_train / (tn_train + fn_train) if (tn_train + fn_train) > 0 else 0
                val_fn_rate = fn_val / (tn_val + fn_val) if (tn_val + fn_val) > 0 else 0
                
                train_aucs.append(train_auc)
                val_aucs.append(val_auc)
                train_f1_5s.append(train_f1_5)
                val_f1_5s.append(val_f1_5)
                train_fn_rates.append(train_fn_rate)
                val_fn_rates.append(val_fn_rate)
            
            # Calculate averages
            avg_train_auc = np.mean(train_aucs)
            avg_val_auc = np.mean(val_aucs)
            avg_train_f1_5 = np.mean(train_f1_5s)
            avg_val_f1_5 = np.mean(val_f1_5s)
            avg_train_fn_rate = np.mean(train_fn_rates)
            avg_val_fn_rate = np.mean(val_fn_rates)
            
            # Store metrics in trial
            trial.set_user_attr("train_auc", avg_train_auc)
            trial.set_user_attr("val_auc", avg_val_auc)
            trial.set_user_attr("train_f1_5", avg_train_f1_5)
            trial.set_user_attr("val_f1_5", avg_val_f1_5)
            trial.set_user_attr("train_fn_rate", avg_train_fn_rate)
            trial.set_user_attr("val_fn_rate", avg_val_fn_rate)
            
            print(f"Train AUC: {avg_train_auc:.4f} - Val AUC: {avg_val_auc:.4f} - Train FN Rate: {avg_train_fn_rate:.4f} - Val FN Rate: {avg_val_fn_rate:.4f}")
            
            return avg_val_auc  # Optimize for validation AUC
            
        except Exception as e:
            print("Failed params:", params)
            print("Error:", e)
            return 0.0 
    
    # Create study
    study = optuna.create_study(
        direction='maximize', 
        sampler=TPESampler(
            seed=42,
            n_startup_trials=3,   
            n_ei_candidates=24   
        )
    )
    
    print("\n" + "="*60)
    print("STARTING LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("="*60)
    print("Starting LR Bayesian Optimization with Optuna...")
    
    study.optimize(enhanced_lr_objective, n_trials=5, show_progress_bar=True)
    
    # Best params
    lr_best_params_bo = study.best_params
    lr_best_score_bo = study.best_value
    
    # Train the model with the best parameters and evaluate it on the test set
    lr_best_model_bo = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear',
        **lr_best_params_bo
    )
    
    lr_best_model_bo.fit(x_train, y_train_lr)
    lr_test_pred_bo = lr_best_model_bo.predict_proba(x_test)[:, 1]
    lr_test_auc_bo = roc_auc_score(y_test, lr_test_pred_bo)
    
    # Results
    print(f"\n=== Logistic Regression Optimization Results ===")
    print(f"Best CV Val AUC: {lr_best_score_bo:.4f}")
    print(f"Test AUC: {lr_test_auc_bo:.4f}")
    print(f"Best Parameters: {lr_best_params_bo}")
    
    # Results for all trials
    print(f"\nDetailed Trial Results:")
    for i, trial in enumerate(study.trials):
        print(f"Trial {i+1}:")
        print(f"  Train AUC: {trial.user_attrs.get('train_auc', 'N/A'):.4f}")
        print(f"  Val AUC: {trial.value:.4f}")
        print(f"  Train F1.5: {trial.user_attrs.get('train_f1_5', 'N/A'):.4f}")
        print(f"  Val F1.5: {trial.user_attrs.get('val_f1_5', 'N/A'):.4f}")
        print(f"  Train FN Rate: {trial.user_attrs.get('train_fn_rate', 'N/A'):.4f}")
        print(f"  Val FN Rate: {trial.user_attrs.get('val_fn_rate', 'N/A'):.4f}")
        print(f"  Params: {trial.params}")
        print()
    
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
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

# Threshold space
thresholds = np.arange(0.1, 0.9, 0.01)
results = []

# Train set prediction probability for LR
lr_train_pred = lr_best_model_bo.predict_proba(x_train)[:, 1]

# Test set prediction probability for LR
lr_test_pred = lr_best_model_bo.predict_proba(x_test)[:, 1]

# Find the best threshold based on F1.5
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

# Find best threshold based on F1.5
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

### Model Saving
print("\n" + "="*60)
print("MODEL SAVING")
print("="*60)

# Save model path
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save LR model, threshold, and scaler
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
        print(f"✅ Scaler saved to: {lr_scaler_path}")
    else:
        lr_scaler_path = "Scaler not found"
        print("⚠️ Warning: Scaler not found in locals()")
    
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
    
    print(f"✅ LogisticRegression model saved to: {lr_joblib_path}")
    print(f"✅ Optimal threshold saved to: {lr_threshold_path}")
    print(f"✅ Model info saved to: {lr_info_path}")
    
else:
    print("❌ LogisticRegression model not found or optimization failed")

### Final Summary
def print_final_summary():
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if 'study' in locals() and study is not None:
        best_trial = study.best_trial
        print(f"📊 LOGISTIC REGRESSION RESULTS:")
        print(f"   Best Val AUC: {best_trial.value:.4f}")
        print(f"   Best Train AUC: {best_trial.user_attrs.get('train_auc', 'N/A'):.4f}")
        print(f"   Best Val F1.5: {best_trial.user_attrs.get('val_f1_5', 'N/A'):.4f}")
        print(f"   Best Train F1.5: {best_trial.user_attrs.get('train_f1_5', 'N/A'):.4f}")
        print(f"   Best Val FN Rate: {best_trial.user_attrs.get('val_fn_rate', 'N/A'):.4f}")
        print(f"   Best Train FN Rate: {best_trial.user_attrs.get('train_fn_rate', 'N/A'):.4f}")
        print(f"   Final Test AUC: {test_auc:.4f}")
        print(f"   Final Test F1.5: {test_f1_5:.4f}")
        print(f"   Final Test FN Rate: {test_fn_rate:.4f}")
        print(f"   Optimal Threshold: {lr_threshold:.4f}")
        print(f"   Best Parameters: {lr_best_params_bo}")
    
    print("="*60)
    print("✅ Logistic Regression training and optimization completed!")

# Call final summary
print_final_summary()
