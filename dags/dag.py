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
    
    # ============= Data Pipeline =============
    dep_check_label_source_data = BashOperator(
        task_id="dep_label_source_check",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/label/dep_label_source_check.py'
        )
    )
    
    dep_check_source_data = BashOperator(
        task_id="dep_check_source_data",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/dep_source_check.py'
        )
    )
    
    # ---- label store ----
    bronze_label_store = DummyOperator(task_id="bronze_label_store")
    silver_label_store = DummyOperator(task_id="silver_label_store")
    gold_label_store = DummyOperator(task_id="gold_label_store")
    label_store_completed = DummyOperator(task_id="label_store_completed")

    # ---- feature store ----
    bronze_member = BashOperator(
        task_id="bronze_member",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/bronze_member.py'
        )    
    )
    silver_member = BashOperator(
        task_id="silver_member",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/silver_member.py'
        )    
    )


    bronze_userlog = BashOperator(
        task_id="bronze_userlog",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/bronze_userlog.py'
        )    
    )
    silver_userlog = DummyOperator(task_id="silver_userlog")

    silver_transaction = DummyOperator(task_id="silver_transaction")
    gold_feature_store = DummyOperator(task_id="gold_feature_store")
    feature_store_completed = DummyOperator(task_id="feature_store_completed")


    # ---- label store ----
    dep_check_source_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # ---- feature store ----
    ### member
    dep_check_source_data >> bronze_member >> silver_member >> gold_feature_store >> feature_store_completed

    ### transaction
    dep_check_source_data >> bronze_label_store >> silver_label_store >> silver_transaction >> gold_feature_store >> feature_store_completed

    ### userlog
    dep_check_source_data >> bronze_userlog >> silver_userlog >> gold_feature_store >> feature_store_completed
    dep_check_source_data >> bronze_label_store >> silver_label_store >> silver_transaction >> silver_userlog >> gold_feature_store >> feature_store_completed
    

    

    # bronze_transaction = BashOperator(
    #     task_id="bronze_transaction",
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 data/bronze_transaction.py'
    #     )
    # )

    

    # silver_transaction = BashOperator(
    #     task_id="silver_transaction",
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 data/silver_transaction.py'
    #     )
    # )

    # silver_userlog = DummyOperator(task_id="silver_userlog")
    # silver_member = DummyOperator(task_id="silver_member")

    # gold_label_store = BashOperator(
    #     task_id="gold_label_store",
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 data/gold_label_store.py'
    #     )
    # )

    # gold_feature_store = DummyOperator(task_id="gold_feature_store")

    # label_store_completed = BashOperator(
    #     task_id="label_store_completed",
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 data/label_store_completed.py'
    #     )
    # )

    # feature_store_completed = DummyOperator(task_id="feature_store_completed")

# ----------------------------------------------------------------------------------------------------------------------------------
    # # Define task dependencies to run scripts sequentially
    # dep_check_source_data >> bronze_transaction >> silver_transaction >> gold_label_store
    
    # # Transaction for Feature Store
    # dep_check_source_data >> bronze_transaction >> silver_transaction >> gold_feature_store

    # # Member for Feature Store
    # dep_check_source_data >> bronze_member >> silver_member >> gold_feature_store

    # # User Log for Feature Store
    # dep_check_source_data >> bronze_transaction >> silver_transaction >> silver_userlog >> gold_feature_store
    # dep_check_source_data >> bronze_userlog >> silver_userlog >> gold_feature_store

    # gold_label_store >> label_store_completed
    # gold_feature_store >> feature_store_completed



    








#=========================#=========================#=========================#=========================#=========================#=========================#=========================


    # # data pipeline

    # # --- label store ---

    # dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    # bronze_label_store = BashOperator(
    #     task_id='run_bronze_label_store',
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 bronze_label_store.py '
    #         '--snapshotdate "{{ ds }}"'
    #     ),
    # )

    # silver_label_store = DummyOperator(task_id="silver_label_store")

    # gold_label_store = DummyOperator(task_id="gold_label_store")

    # label_store_completed = DummyOperator(task_id="label_store_completed")

    # # Define task dependencies to run scripts sequentially
    # dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 
 
    # # --- feature store ---
    # dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")

    # dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_bronze_2")

    # dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_bronze_3")

    # bronze_table_1 = DummyOperator(task_id="bronze_table_1")
    
    # bronze_table_2 = DummyOperator(task_id="bronze_table_2")

    # bronze_table_3 = DummyOperator(task_id="bronze_table_3")

    # silver_table_1 = DummyOperator(task_id="silver_table_1")

    # silver_table_2 = DummyOperator(task_id="silver_table_2")

    # gold_feature_store = DummyOperator(task_id="gold_feature_store")

    # feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # # Define task dependencies to run scripts sequentially
    # dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    # dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    # dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    # gold_feature_store >> feature_store_completed


    # # --- model inference ---
    # model_inference_start = DummyOperator(task_id="model_inference_start")

    # model_1_inference = DummyOperator(task_id="model_1_inference")

    # model_2_inference = DummyOperator(task_id="model_2_inference")

    # model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # # Define task dependencies to run scripts sequentially
    # feature_store_completed >> model_inference_start
    # model_inference_start >> model_1_inference >> model_inference_completed
    # model_inference_start >> model_2_inference >> model_inference_completed


    # # --- model monitoring ---
    # model_monitor_start = DummyOperator(task_id="model_monitor_start")

    # model_1_monitor = DummyOperator(task_id="model_1_monitor")

    # model_2_monitor = DummyOperator(task_id="model_2_monitor")

    # model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # # Define task dependencies to run scripts sequentially
    # model_inference_completed >> model_monitor_start
    # model_monitor_start >> model_1_monitor >> model_monitor_completed
    # model_monitor_start >> model_2_monitor >> model_monitor_completed


    # # --- model auto training ---

    # model_automl_start = DummyOperator(task_id="model_automl_start")
    
    # model_1_automl = DummyOperator(task_id="model_1_automl")

    # model_2_automl = DummyOperator(task_id="model_2_automl")

    # model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # # Define task dependencies to run scripts sequentially
    # feature_store_completed >> model_automl_start
    # label_store_completed >> model_automl_start
    # model_automl_start >> model_1_automl >> model_automl_completed
    # model_automl_start >> model_2_automl >> model_automl_completed