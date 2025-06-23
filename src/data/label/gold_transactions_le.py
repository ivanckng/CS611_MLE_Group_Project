import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.gold

bucket_name = "cs611_mle"

def main():
    print('\n\n---Starting Job ~ Gold Transactions (For Label)---\n\n')

    silver_transactions_file_path = "Silver Layer/silver_transactions.csv"
    gold_transactions_file_path = "Gold Layer/labels.csv"

    grace_period = 5
    utils.gold.process_gold_label_store(bucket_name, silver_transactions_file_path, gold_transactions_file_path, grace_period)
    
    print('\n\n---Done ~ Gold Transactions Table (For Label)---\n\n')

main()