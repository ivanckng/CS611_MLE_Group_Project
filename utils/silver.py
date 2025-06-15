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


def process_silver_table_member(bucket_name, bronze_member_directory, silver_member_directory, spark):

    # connect to bronze table
    partition_name = "bronze_members.csv"
    filepath = f"gs://{bucket_name}/{bronze_member_directory}/{partition_name}"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Drop unnamed 0, invalid column
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
    filepath = f"gs://{bucket_name}/{silver_member_directory}/{partition_name}"
    try:
        df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
        return df
    except Exception as e:
        print(f'failed to save silver member: {e}')
        return None


def process_silver_table_transaction(date_str, bucket_name, bronze_transaction_directory, finalized_silver_transaction_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_transaction_" + date_str.replace('-','_') + '.csv'
    filepath = f"gs://{bucket_name}/{bronze_transaction_directory}/{partition_name}"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())


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

    # valid_range = list(range(36, 42))
    # df = ( df .withColumn( "payment_method_id", F.when(F.col("payment_method_id").isin(valid_range), F.col("payment_method_id").cast("string")) .otherwise(F.lit("Others")) ) )


    df = ( df .withColumn( "discount_ratio", F.when( F.col("plan_list_price") != 0, (F.col("plan_list_price") - F.col("actual_amount_paid")) / F.col("plan_list_price") ).otherwise(F.lit(0)) ) )



    # save silver table - IRL connect to database to write
    partition_name = "silver_transaction_" + date_str.replace('-','_') + '.parquet'
    filepath = f"gs://{bucket_name}/{finalized_silver_transaction_directory}/{partition_name}"
    try:
        df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
        return df
    except Exception as e:
        print(f'failed to save silver member: {e}')
        return None
    




def process_silver_table_userlog(date_str, bucket_name, bronze_userlog_directory, silver_transaction_directory, silver_userlog_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Current Month Partition UserLog
    partition_name = "bronze_userlog_" + date_str.replace('-','_') + '.csv'
    filepath = f"gs://{bucket_name}/{bronze_userlog_directory}/{partition_name}"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Prepare next-month snapshot
    next_month_date = snapshot_date + relativedelta(months=1)
    next_date_str = next_month_date.strftime("%Y-%m-%d")
    next_partition_name = "bronze_userlog_" + next_date_str.replace('-','_') + '.csv'
    next_filepath = f"gs://{bucket_name}/{bronze_userlog_directory}/{next_partition_name}"

    # Try reading the next snapshot, if exists
    try:
        df_next = spark.read.csv(next_filepath, header=True, inferSchema=True)
        print('Also loaded from:', next_filepath, 'row count:', df_next.count())
        df = df.unionByName(df_next)
    except AnalysisException:
        print(f"File {next_filepath} does not exist. Proceeding with single snapshot.")


    # Drop unnamed 0, invalid column
    df = df.drop('Unnamed: 0')


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
    # previous_month = (snapshot_date - relativedelta(months=1)).replace(day=1)

    # f"gs://{bucket_name}/{bronze_userlog_directory}/{partition_name}"
    
    # Load current Month
    tx_current_partition = "silver_transaction_" + current_month.strftime("%Y_%m_%d") + ".parquet" 
    tx_current_path = f"gs://{bucket_name}/{silver_transaction_directory}/{tx_current_partition}"
    df_tx_current = spark.read.parquet(tx_current_path)


    # if previous_month >= min_tx_date:
    #     tx_prev_partition = "silver_transaction_" + previous_month.strftime("%Y_%m_%d") + ".parquet"
    #     tx_prev_path = f"gs://{bucket_name}/{silver_transaction_directory}/{tx_prev_partition}"
    #     df_tx_prev = spark.read.parquet(tx_prev_path)
    #     df_tx = df_tx_current.unionByName(df_tx_prev)
    #     print('Loaded transactions from:', tx_prev_partition, 'and', tx_current_partition, 'row count:', df_tx.count())
    # else:
    df_tx = df_tx_current
    #     print('Loaded transactions only from:', tx_current_partition, 'row count:', df_tx.count())
    

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
    # USER LOG CONTAINS CURRENT MONTH + NEXT MONTH
    # E.G. Tx membership start date = 15-01-2015, membership end date = 15-02-2015
    # Need to get userlog between 01-02-2015

    # ALSO, if last_first_7_days_total_secs_ratio is 0, most likely:
    # 1. the plan is more than 30/31 days, i could not capture more than 2 months of user logs
    # 2. no engagement last 7 days, "total_secs_last_7_days" == 0
    # ======================


    # save silver table - IRL connect to database to write
    partition_name = "silver_userlog_" + date_str.replace('-','_') + '.parquet'
    filepath = f"gs://{bucket_name}/{silver_userlog_directory}/{partition_name}"
    try:
        agg_df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
        return agg_df
    except Exception as e:
        print(f'failed to save silver member: {e}')
        return None

    