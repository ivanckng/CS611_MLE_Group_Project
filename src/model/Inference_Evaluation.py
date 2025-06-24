import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Read data(inference prediction & real)
bucket_name = "cs611_mle"
infer_path_in_bucket = "Gold Layer/prediction.csv"
infer_gcs_path = f"gs://{bucket_name}/{infer_path_in_bucket}"

df = pd.read_csv(infer_gcs_path)

#y_pred
y_pred = df.prediction
#y_true
y_true = df.churn

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
