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


def process_gold_featurestore(date_str, bucket_name, actual_silver_transaction_directory, silver_userlog_directory, silver_member_directory, gold_feature_store_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    
    # =========== 修改成你的 GCS 路径 ============
    transaction_parquet_path = f"gs://{bucket_name}/{actual_silver_transaction_directory}/silver_transaction_{date_str.replace('-','_')}.parquet"
    userlog_parquet_path = f"gs://{bucket_name}/{silver_userlog_directory}/silver_userlog_{date_str.replace('-','_')}.parquet"
    member_parquet_path = f"gs://{bucket_name}/{silver_member_directory}/silver_member.parquet"
    
    
    # ======== 读取 Parquet 文件为 DataFrame ========
    df_transaction = spark.read.parquet(transaction_parquet_path)
    print('loaded from:', transaction_parquet_path, 'row count:', df_transaction.count())
    df_userlog = spark.read.parquet(userlog_parquet_path)
    print('loaded from:', userlog_parquet_path, 'row count:', df_userlog.count())
    df_member = spark.read.parquet(member_parquet_path)
    print('loaded from:', member_parquet_path, 'row count:', df_member.count())

    # df_transaction.show(5)
    # df_userlog.show(5)
    # df_member.show(5)


    # ==== Your join logic here...
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

    # df_joined.show(5)


    partition_name = "gold_featurestore_" + date_str.replace('-','_') + '.parquet'
    filepath = f"gs://{bucket_name}/{gold_feature_store_directory}/{partition_name}"
    try:
        df_joined.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
        return df_joined
    except Exception as e:
        print(f'failed to save gold featurestore: {e}')
        return None
