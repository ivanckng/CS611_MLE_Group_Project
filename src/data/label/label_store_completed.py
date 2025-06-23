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


print("Checking Label Data Source...")

bucket_name = "cs611_mle"
src_directory = "Gold Layer/labels.csv"
label_gcs_path = f"gs://{bucket_name}/{src_directory}"

try:
    df_transactions = pd.read_csv(label_gcs_path)
    print('\n\n---Label Source Exists---\n\n')
except FileNotFoundError:
    print(f"Label Data Source '{label_gcs_path}' does not exist. Please check the label data source.")
    raise SystemExit("Exiting the program due to missing label data source.")
except Exception as e:
    print(f"Error reading from GCS: {e}")

