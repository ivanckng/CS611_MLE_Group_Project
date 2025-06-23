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

import utils.silver

print('Done Importing!')





def main():
    print('\n\n---starting job ~ Silver Member---\n\n')

    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")  
    
    bronze_member_directory = "datamart/bronze/member"
    silver_member_directory = "datamart/silver/member"
    if not os.path.exists(silver_member_directory):
        os.makedirs(silver_member_directory)

    utils.silver.process_silver_table_member(bronze_member_directory, silver_member_directory, spark)

    # end spark session
    spark.stop()
    
    print('\n\n---Done Build Silver Table - Member---\n\n')



main()