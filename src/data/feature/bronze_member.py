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

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import utils.bronze
print('Done Importing!')




def main():
    print('\n\n---starting job ~ Bronze Member---\n\n')

    # Create a Spark session
    spark = SparkSession \
        .builder \
        .config("spark.jars", "jars/gcs-connector-hadoop3-2.2.20-shaded.jar") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    spark._jsc.hadoopConfiguration().set('fs.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem')
    spark._jsc.hadoopConfiguration().set('fs.gs.auth.service.account.enable', 'true')
    spark._jsc.hadoopConfiguration().set('google.cloud.auth.service.account.json.keyfile', "application_default_credentials.json")  
    
    bronze_member_directory = "datamart/bronze/member"
    # if not os.path.exists(bronze_member_directory):
    #     os.makedirs(bronze_member_directory)

    utils.bronze.process_bronze_member(bronze_member_directory, spark)

    # end spark session
    spark.stop()
    
    print('\n\n---Done Build Bronze Table - Member---\n\n')
    



main()

