import numpy as np
import pandas as pd

import utils.silver

bucket_name = "cs611_mle"

def main():
    print('\n\n---Starting Job ~ Silver Transactions (For Label)---\n\n')

    bronze_transactions_file_path = "Bronze Layer/bronze_transactions.csv"
    silver_transactions_file_path = "Silver Layer/silver_transactions.parquet"

    utils.silver.process_silver_transactions_le(bucket_name, bronze_transactions_file_path, silver_transactions_file_path)
    
    print('\n\n---Done ~ Silver Transactions Table (For Label)---\n\n')

main()