from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run daily',
    schedule_interval='0 0 * * *', # run daily
    start_date=datetime(2017, 1, 1),
    end_date=datetime(2017, 3, 31),
    catchup=True,
) as dag:
    
    # # ============= Data Pipeline =============

    # ~~~~~~~ Tasks Definition ~~~~~~~
    ## ---- data check ----
    dep_label_source_check = BashOperator(
        task_id="dep_label_source_check",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/label/dep_label_source_check.py'
        )
    )
    
    dep_feature_source_check = BashOperator(
        task_id="dep_feature_source_check",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/dep_feature_source_check.py'
        )
    )
    
    ## ---- label store ----
    bronze_label_store = BashOperator(
        task_id="bronze_label_store",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/label/bronze_transactions_le.py'
        )
    )

    silver_label_store = BashOperator(
        task_id="silver_label_store",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/label/silver_transactions_le.py'
        )
    )

    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/label/gold_transactions_le.py'
        )
    )

    label_store_completed = BashOperator(
        task_id="label_store_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/label/label_store_completed.py'
        )
    )

    ## ---- feature store ----
    ### member feature store
    bronze_member = BashOperator(
        task_id="bronze_member",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/bronze_member.py'
        )    
    )
    silver_member = BashOperator(
        task_id="silver_member",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/silver_member.py'
        )    
    )

    ### userlog feature store
    # bronze_userlog = BashOperator(
    #     task_id="bronze_userlog",
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 data/feature/bronze_userlog.py'
    #     )    
    # )
    bronze_userlog = DummyOperator(task_id="bronze_userlog")
    silver_userlog = DummyOperator(task_id="silver_userlog")

    ### transaction feature store
    bronze_transaction = DummyOperator(task_id="bronze_transaction")
    silver_transaction = DummyOperator(task_id="silver_transaction")


    ### gold feature store
    gold_feature_store = DummyOperator(task_id="gold_feature_store")
    feature_store_completed = DummyOperator(task_id="feature_store_completed")



    # ~~~~~~~ Flow of tasks ~~~~~~~
    # ---- label store ----
    dep_label_source_check >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # ---- feature store ----
    ### member
    dep_feature_source_check >> bronze_member >> silver_member >> gold_feature_store >> feature_store_completed

    ### transaction
    dep_feature_source_check >> bronze_transaction >> silver_transaction >> gold_feature_store >> feature_store_completed

    ### userlog
    dep_feature_source_check >> bronze_userlog >> silver_userlog >> gold_feature_store >> feature_store_completed
    dep_feature_source_check >> bronze_transaction >> silver_transaction >> silver_userlog >> gold_feature_store >> feature_store_completed