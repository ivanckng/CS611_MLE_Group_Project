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

from pyspark.sql.functions import col, to_date, count, min, max, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_member(bronze_member_directory, spark):
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data_source/members_50k.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print('row count:', row_count)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_members.csv"
    filepath = f"{bronze_member_directory}/{partition_name}"
    try:
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)
        return df
    except Exception as e:
        print(f'failed to save bronze member: {e}')
        return None



def process_bronze_userlog(date_str, bucket_name, bronze_userlog_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Get the actual last day of the month
    year = snapshot_date.year
    month = snapshot_date.month
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day)
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = f"gs://{bucket_name}/Bronze Layer/user_logs_50k.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    df = df.withColumn('date', to_date(col('date').cast('string'), 'yyyyMMdd'))
    df = df.filter( (col("date") >= lit(snapshot_date)) & (col("date") <= lit(end_date)) )
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print(date_str + 'row count:', row_count)


    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_userlog_" + date_str.replace('-','_') + '.csv'
    filepath = f"gs://{bucket_name}/{bronze_userlog_directory}/{partition_name}"
    try:
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)
        return df
    except Exception as e:
        print(f'failed to save bronze userlog {date_str}: {e}')
        return None


def process_bronze_transaction_partition(date_str, bucket_name, bronze_transaction_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Get the actual last day of the month
    year = snapshot_date.year
    month = snapshot_date.month
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day)
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = f"gs://{bucket_name}/Silver Layer/silver_transactions.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    df = df.withColumn('membership_start_date', to_date(col('membership_start_date').cast('string'), 'yyyy-MM-dd'))
    df = df.filter( (col("membership_start_date") >= lit(snapshot_date)) & (col("membership_start_date") <= lit(end_date)) )
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print(date_str + 'row count:', row_count)


    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_transaction_" + date_str.replace('-','_') + '.csv'
    filepath = f"gs://{bucket_name}/{bronze_transaction_directory}/{partition_name}"
    try:
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)
        return df
    except Exception as e:
        print(f'failed to save bronze transaction {date_str}: {e}')
        return None