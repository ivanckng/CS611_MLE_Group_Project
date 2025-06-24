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


def process_silver_table_member(bronze_member_directory, silver_member_directory, spark):

    # connect to bronze table
    partition_name = "bronze_members.csv"
    filepath = f"gs://cs611_mle/{bronze_member_directory}/{partition_name}"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Drop invalid column
    df = df.drop('_c0')
    df = df.drop('Unnamed: 0')
    df = df.drop('bd') # lots of invalid data through EDA
    


    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "msno":StringType(),
        "city":StringType(),
        "gender":StringType(),
        "registered_via":StringType()
    }

    # cast to their respective data types
    # withColumn(colName: str, col: pyspark.sql.column.Column) 
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    df = df.withColumn('registration_init_time', to_date(col('registration_init_time').cast('string'), 'yyyyMMdd'))

    # Lots of missing 'gender' data through EDA
    df.groupBy('gender').count().orderBy('count', ascending=False).show()
    df = df.fillna({'gender': 'na'})

    


    # save silver table - IRL connect to database to write
    partition_name = 'silver_member.parquet'
    filepath = f"gs://cs611_mle/{silver_member_directory}/{partition_name}"
    try:
        df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath, 'row count:', df.count())
        df.show(5)  # Show the first 5 rows for verification
        return df
    except Exception as e:
        print(f'failed to save silver member: {e}')
        return None


def process_silver_table_transaction(date_str, bronze_transaction_directory, silver_transaction_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_transaction_" + date_str.replace('-','_') + '.csv'
    filepath = f"gs://cs611_mle/{bronze_transaction_directory}/{partition_name}"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # drop invalid columns
    df = df.drop('Unnamed: 0')
    df = df.drop('_c0')

    df_transactions = df.toPandas()
    df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'], format='%Y%m%d')
    df_transactions['membership_expire_date'] = pd.to_datetime(df_transactions['membership_expire_date'], format='%Y-%m-%d')

    
    # Add start date column
    df_transactions['membership_start_date'] = (
        df_transactions['membership_expire_date'] -
        pd.to_timedelta(df_transactions['payment_plan_days'], unit='D')
    )
    
    df_transactions = df_transactions.sort_values(by=['msno', 'transaction_date', 'membership_expire_date'])

    # Filter outliers
    daily_transactions_by_user = df_transactions.groupby(['msno', 'transaction_date', 'is_cancel']).size().reset_index(name='transaction_count')
    daily_transactions_by_user = daily_transactions_by_user.sort_values(by = 'transaction_count', ascending = False)

    outlier_days = 2
    outlier_user_list = list(daily_transactions_by_user[daily_transactions_by_user['transaction_count'] > outlier_days]['msno'].unique())
    filtering = lambda x: True if x not in outlier_user_list else False
    df_transactions['filtered'] = df_transactions['msno'].apply(filtering)

    df_transactions = df_transactions[df_transactions['filtered'] == True]
    df_transactions = df_transactions.drop(columns=['filtered'])

    # Filter date
    start_date = '2015-01-01'
    end_date = '2017-03-31'
    df_transactions = df_transactions[df_transactions['membership_start_date'] >= start_date]
    df_transactions = df_transactions[df_transactions['membership_expire_date'] <= end_date]

    df_transactions = df_transactions.reset_index(drop=True)
    # Convert back to Spark DataFrame
    df = spark.createDataFrame(df_transactions)
    
    


    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "msno": StringType(),
        "payment_method_id": StringType(),
        "payment_plan_days":IntegerType(),
        "plan_list_price": FloatType(),
        "actual_amount_paid":FloatType(),
        "is_auto_renew": IntegerType(),
        "transaction_date": DateType(),
        "membership_expire_date": DateType(),
        "is_cancel": IntegerType(),
        "membership_start_date": DateType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))


    df = ( df .withColumn( "discount_ratio", F.when( F.col("plan_list_price") != 0, (F.col("plan_list_price") - F.col("actual_amount_paid")) / F.col("plan_list_price") ).otherwise(F.lit(0)) ) )



    # save silver table - IRL connect to database to write
    partition_name = "silver_transaction_" + date_str.replace('-','_') + '.parquet'
    filepath = f"gs://cs611_mle/{silver_transaction_directory}/{partition_name}"
    try:
        df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath, 'row count:', df.count())
        df.show(5)  # Show the first 5 rows for verification
        return df
    except Exception as e:
        print(f'failed to save silver member: {e}')
        return None
    




def process_silver_table_userlog(date_str, bronze_userlog_directory, silver_transaction_directory, silver_userlog_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Current Month Partition UserLog
    partition_name = "bronze_userlog_" + date_str.replace('-','_') + '.csv'
    filepath = f"gs://cs611_mle/{bronze_userlog_directory}/{partition_name}"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Prepare last-month snapshot
    last_month_date = snapshot_date - relativedelta(months=1)
    last_date_str = last_month_date.strftime("%Y-%m-%d")
    last_partition_name = "bronze_userlog_" + last_date_str.replace('-', '_') + '.csv'
    last_filepath = f"gs://cs611_mle/{bronze_userlog_directory}/{last_partition_name}"

    # Try reading the last snapshot, if exists
    try:
        df_last = spark.read.csv(last_filepath, header=True, inferSchema=True)
        print('Also loaded from:', last_filepath, 'row count:', df_last.count())
        df = df.unionByName(df_last)
    except AnalysisException:
        print(f"File {last_filepath} does not exist. Proceeding with single snapshot.")


    # Drop invalid column
    df = df.drop('_c0')
    df = df.drop('Unnamed: 0')

    # Define the bounds
    lower_bound = -0.75 * 1e16
    upper_bound = 0.75 * 1e16
    
    # Filter out invalid rows
    df = df.filter((col("total_secs") >= lower_bound) & (col("total_secs") <= upper_bound))


    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "msno": StringType(),
        "date":DateType(),
        "num_25":IntegerType(),
        "num_50":IntegerType(),
        "num_75":IntegerType(),
        "num_985":IntegerType(),
        "num_100":IntegerType(),
        "num_unq":IntegerType(),
        "total_secs":FloatType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))


    # ============ GET TRANSACTION ============
    min_tx_date = datetime.strptime("2015-01-01", "%Y-%m-%d")
    
    # Get current and previous month (always use 1st of month)
    current_month = snapshot_date.replace(day=1)
    
    # Load current Month
    tx_current_partition = "silver_transaction_" + current_month.strftime("%Y_%m_%d") + ".parquet" 
    tx_current_path = f"gs://cs611_mle/{silver_transaction_directory}/{tx_current_partition}"
    df_tx_current = spark.read.parquet(tx_current_path)
    print('Loaded transactions from:', tx_current_partition, 'row count:', df_tx_current.count())


    df_tx = df_tx_current
    
    # JOIN logs and transaction
    df_joined = df.join(
        df_tx,
        on=(
            (df.msno == df_tx.msno) &
            (df.date >= df_tx.membership_start_date) &
            (df.date <= df_tx.membership_expire_date)
        ),
        how='inner'
    )
    
    # Drop the duplicate msno column from df_tx
    df_joined = df_joined.drop(df_tx.msno)

    # ========== Feature Engineering ==========
    # sum_songs each log
    df_joined = df_joined.withColumn(
        "sum_songs", col("num_25") + col("num_50") + col("num_75") + col("num_985") + col("num_100")
    )
    
    # Unique song played ratio
    df_joined = df_joined.withColumn(
        "unique_song_played_ratio",
        when(col("sum_songs") == 0, 0).otherwise(col("num_unq") / col("sum_songs"))
    )


    # first 7 days vs last 7 days
    df_joined = df_joined.withColumn(
        "days_from_start", datediff("date", "membership_start_date")
    ).withColumn(
        "days_to_end", datediff("membership_expire_date", "date")
    )


    

    # Aggregate
    agg_df = df_joined.groupBy(
        "msno", "membership_start_date", "membership_expire_date", "payment_method_id",
        "payment_plan_days", "plan_list_price", "actual_amount_paid", "is_auto_renew",
        "transaction_date", "is_cancel"
    ).agg(
        spark_sum("num_25").alias("sum_completion_25"),
        spark_sum("num_50").alias("sum_completion_50"),
        spark_sum("num_75").alias("sum_completion_75"),
        spark_sum("num_985").alias("sum_completion_985"),
        spark_sum("num_100").alias("sum_completion_100"),
        spark_sum("sum_songs").alias("sum_songs"),
        spark_sum("total_secs").alias("sum_total_secs"),

        
        spark_avg("unique_song_played_ratio").alias('avg_unique_song_played_ratio'),
        coalesce(spark_sum(when(col("days_from_start") < 7, col("total_secs"))), lit(0)).alias("total_secs_first_7_days"),
        coalesce(spark_sum(when(col("days_to_end") < 7, col("total_secs"))), lit(0)).alias("total_secs_last_7_days")
    )

    columns_to_drop = [
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "is_cancel"
    ] # Exclude these 
    agg_df = agg_df.drop(*columns_to_drop)


    # Average seconds per song
    agg_df = agg_df.withColumn(
        "avg_seconds_per_songs",
        when(col("sum_songs") == 0, 0).otherwise(col("sum_total_secs") / col("sum_songs"))
    )

    # Skip ratio (25% + 50%)
    agg_df = agg_df.withColumn(
        "skip_ratio",
        when(col("sum_songs") == 0, 0).otherwise((col("sum_completion_25") + col("sum_completion_50")) / col("sum_songs"))
    )


    # Songs played ratio (25%)
    agg_df = agg_df.withColumn(
        "songs_played_25_ratio",
        when(col("sum_songs") == 0, 0).otherwise(col("sum_completion_25") / col("sum_songs"))
    )

    # Songs played ratio (50%)
    agg_df = agg_df.withColumn(
        "songs_played_50_ratio",
        when(col("sum_songs") == 0, 0).otherwise(col("sum_completion_50") / col("sum_songs"))
    )

    # Songs played ratio (75%)
    agg_df = agg_df.withColumn(
        "songs_played_75_ratio",
        when(col("sum_songs") == 0, 0).otherwise(col("sum_completion_75") / col("sum_songs"))
    )

    # Songs played ratio (75%)
    agg_df = agg_df.withColumn(
        "songs_played_985_ratio",
        when(col("sum_songs") == 0, 0).otherwise(col("sum_completion_985") / col("sum_songs"))
    )

    # Songs completion ratio (100%)
    agg_df = agg_df.withColumn(
        "songs_completion_ratio",
        when(col("sum_songs") == 0, 0).otherwise(col("sum_completion_100") / col("sum_songs"))
    )


    # first 7 days vs last 7 days ratio
    agg_df = agg_df.withColumn(
        "last_first_7_days_total_secs_ratio",
        when(col("total_secs_last_7_days") == 0, 0)
        .otherwise(
            (col("total_secs_last_7_days") + 0.01) / (col("total_secs_first_7_days") + 0.01)
        )
    )


    

    # ====================
    # USER LOG CONTAINS CURRENT MONTH + LAST MONTH
    # E.G. Tx membership expire date = 15-02-2015, membership start date = 15-01-2015
    # Need to get userlog between 01-2025 and 02-2015

    # ALSO, if last_first_7_days_total_secs_ratio is 0, most likely:
    # 1. the plan is more than 30/31 days, i could not capture more than 2 months of user logs
    # 2. no engagement last 7 days, "total_secs_last_7_days" == 0
    # ======================


    # save silver table - IRL connect to database to write
    partition_name = "silver_userlog_" + date_str.replace('-','_') + '.parquet'
    filepath = f"gs://cs611_mle/{silver_userlog_directory}/{partition_name}"
    try:
        agg_df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath, 'row count:', agg_df.count())
        agg_df.show(5)  # Show the first 5 rows for verification
        return agg_df
    except Exception as e:
        print(f'failed to save silver member: {e}')
        return None

def process_silver_transactions_le(bucket_name, src_directory, target_directory):
    # Load data
    bronze_transactions_gcs_path = f"gs://{bucket_name}/{src_directory}"
    df_transactions = pd.read_csv(bronze_transactions_gcs_path)

    # Preprocessing
    df_transactions = df_transactions.drop(columns=['Unnamed: 0'])
    df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'], format='%Y%m%d')
    df_transactions['membership_expire_date'] = pd.to_datetime(df_transactions['membership_expire_date'], format='%Y%m%d')
    
    # Add start date column
    df_transactions['membership_start_date'] = (
        df_transactions['membership_expire_date'] -
        pd.to_timedelta(df_transactions['payment_plan_days'], unit='D')
    )
    
    df_transactions = df_transactions.sort_values(by=['msno', 'transaction_date', 'membership_expire_date'])

    # Filter outliers
    daily_transactions_by_user = df_transactions.groupby(['msno', 'transaction_date', 'is_cancel']).size().reset_index(name='transaction_count')
    daily_transactions_by_user = daily_transactions_by_user.sort_values(by = 'transaction_count', ascending = False)

    outlier_days = 2
    outlier_user_list = list(daily_transactions_by_user[daily_transactions_by_user['transaction_count'] > outlier_days]['msno'].unique())
    filtering = lambda x: True if x not in outlier_user_list else False
    df_transactions['filtered'] = df_transactions['msno'].apply(filtering)

    df_transactions = df_transactions[df_transactions['filtered'] == True]
    df_transactions = df_transactions.drop(columns=['filtered'])

    # Filter date
    start_date = '2015-01-01'
    end_date = '2017-03-31'
    df_transactions = df_transactions[df_transactions['membership_start_date'] >= start_date]
    df_transactions = df_transactions[df_transactions['membership_expire_date'] <= end_date]

    df_transactions = df_transactions.reset_index(drop=True)

    # Add 'next_transaction_date' and 'days_diff' columns
    result_rows = []

    grouped = df_transactions.groupby('msno', sort=False)
    for msno, group in grouped:
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            row = group.iloc[i]

            start_date = row['membership_start_date']
            expire_date = row['membership_expire_date']
            payment_plan_days = row['payment_plan_days']

            if 30 <= payment_plan_days <= 31 and row['is_cancel'] == 0:

                next_transaction_date = None
                days_diff = None
                for j in range(i + 1, len(group)):
                    next_row = group.iloc[j]
                    if next_row['is_cancel'] == 0:
                        next_transaction_date = next_row['transaction_date']
                        days_diff = (next_transaction_date - expire_date).days
                        break

                result_rows.append({
                    'msno': row['msno'],
                    'membership_start_date': start_date,
                    'membership_expire_date': expire_date,
                    'next_transaction_date': next_transaction_date,
                    'days_diff': days_diff,
                })

    df_result = pd.DataFrame(result_rows)

    df_result.loc[df_result['days_diff'] < 0, 'days_diff'] = 0

    # Check for small gaps between "membership_expire_date" and "2017-03-31", and if "next_transaction_date" is None,
    # drop these rows because it's still unknown whether there will be a next transaction after this or not.
    mask_drop = (
        df_result['next_transaction_date'].isna() &
        (pd.to_datetime('2017-03-31') - df_result['membership_expire_date']).dt.days.between(0, 5)
    )
    df_result = df_result[~mask_drop]

    # Create silver table and store to Google Cloud Storage
    silver_transactions_gcs_path = f"gs://{bucket_name}/{target_directory}"
    try:
        df_result.to_parquet(silver_transactions_gcs_path, index=False)
        print("silver_transactions.csv Stored to Silver Layer Successfully! ✅")
    except Exception as e:
        print(f"silver_transactions.csv Store Failed: {e}")