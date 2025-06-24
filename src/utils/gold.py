import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
import calendar

from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from pyspark.sql.functions import col, sum as spark_sum, countDistinct, when, datediff, expr, row_number, to_date, count, min, max, lit, avg as spark_avg, coalesce

from pyspark.sql.window import Window

from pyspark.sql.utils import AnalysisException


def process_gold_featurestore(date_str, silver_transaction_directory, silver_userlog_directory, silver_member_directory, gold_feature_store_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    
    # =========== 修改成你的 GCS 路径 ============
    transaction_parquet_path = f"gs://cs611_mle/{silver_transaction_directory}/silver_transaction_{date_str.replace('-','_')}.parquet"
    userlog_parquet_path = f"gs://cs611_mle/{silver_userlog_directory}/silver_userlog_{date_str.replace('-','_')}.parquet"
    member_parquet_path = f"gs://cs611_mle/{silver_member_directory}/silver_member.parquet"
    
    
    # ======== 读取 Parquet 文件为 DataFrame ========
    df_transaction = spark.read.parquet(transaction_parquet_path)
    print('loaded from:', transaction_parquet_path, 'row count:', df_transaction.count())
    df_userlog = spark.read.parquet(userlog_parquet_path)
    print('loaded from:', userlog_parquet_path, 'row count:', df_userlog.count())
    df_member = spark.read.parquet(member_parquet_path)
    print('loaded from:', member_parquet_path, 'row count:', df_member.count())


    df_joined = df_transaction.join(
        df_userlog,
        on=['msno', 'membership_start_date', 'membership_expire_date'],
        how='inner'  # or 'left', 'right', 'outer' depending on your requirement
    )
    
    df_joined = df_joined.join(
        df_member,
        on='msno',
        how='inner'  # adjust the join type if needed
    )

    # account age
    df_joined = df_joined.withColumn(
        "account_age",
        datediff("membership_start_date", "registration_init_time")
    )

    # drop not needed columns
    df_joined = df_joined.drop('plan_list_price')
    df_joined = df_joined.drop('actual_amount_paid')
    df_joined = df_joined.drop('payment_plan_days')
    df_joined = df_joined.drop('registration_init_time')
    df_joined = df_joined.drop('registration_init_time')


    partition_name = "gold_featurestore_" + date_str.replace('-','_') + '.parquet'
    filepath = f"gs://cs611_mle/{gold_feature_store_directory}/{partition_name}"
    try:
        df_joined.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath, 'row count:', df_joined.count())
        df_joined.show(5)
        return df_joined
    except Exception as e:
        print(f'failed to save gold featurestore: {e}')
        return None

def process_gold_label_store(bucket_name, src_directory, target_directory, grace_period):
    # Ingesting data from Google Cloud Storage
    silver_transactions_gcs_path = f"gs://{bucket_name}/{src_directory}"
    df_transactions = pd.read_parquet(silver_transactions_gcs_path)

    # Generate labels
    grace_period = 5
    df_transactions['churn'] = ((df_transactions['days_diff'] > grace_period) | (df_transactions['days_diff'] == "None")).astype(int)

    df_labels = df_transactions[['msno', 'membership_start_date', 'membership_expire_date', 'churn']]

    # Create gold table and store to Google Cloud Storage
    gold_label_gcs_path = f"gs://{bucket_name}/{target_directory}"
    try:
        df_labels.to_csv(gold_label_gcs_path, index=False)
        print("labels.csv Stored to Gold Layer Successfully! ")
    except Exception as e:
        print(f"labels.csv Store Failed: {e}")