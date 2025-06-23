import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.silver

bucket_name = "cs611_mle"

def main():
    print('\n\n---Starting Job ~ Silver Transactions (For Label)---\n\n')

    bronze_transactions_file_path = "Bronze Layer/bronze_transactions.csv"
    silver_transactions_file_path = "Silver Layer/silver_transactions.csv"

    utils.silver.process_silver_transactions_le(bucket_name, bronze_transactions_file_path, silver_transactions_file_path)
    
    print('\n\n---Done ~ Silver Transactions Table (For Label)---\n\n')

main()