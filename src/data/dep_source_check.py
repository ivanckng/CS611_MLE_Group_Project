import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from pyspark.sql import SparkSession
from tqdm import tqdm
import time  # to simulate loading for tqdm
import sys
import os
import glob
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, to_date, count, min, max, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, sum as spark_sum, when

print('Done Importing!')


spark = SparkSession \
    .builder \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()


def dependency_check(path, file_name):
    # load data - IRL ingest from back end source system
    df = spark.read.csv(path+file_name, header=True, inferSchema=True)
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print(f'{file_name} row count:', row_count)


file_names = ['members_50k.csv', 'transactions_50k.csv', 'user_logs_50k.csv']
# connect to source back end - IRL connect to back end source system
file_path = "../data_source/"

for file_name in file_names:
    print(f'Checking {file_name}...')
    dependency_check(file_path, file_name)
    print(f'{file_name} check completed.\n')
