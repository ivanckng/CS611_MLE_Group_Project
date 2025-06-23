import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.bronze

bucket_name = "cs611_mle"

def main():
    print('\n\n---Starting Job ~ Bronze Transactions (For Label)---\n\n')

    transactions_csv_file_path = "Data Source/transactions_50k.csv"
    bronze_transactions_file_path = "Bronze Layer/bronze_transactions.csv"

    utils.bronze.process_bronze_transactions_le(bucket_name, transactions_csv_file_path, bronze_transactions_file_path)
    
    print('\n\n---Done ~ Bronze Transactions Table (For Label)---\n\n')

main()