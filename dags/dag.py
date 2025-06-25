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
    bronze_userlog = BashOperator(
        task_id="bronze_userlog",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/bronze_userlog.py'
        )    
    )
    silver_userlog = BashOperator(
        task_id="silver_userlog",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/silver_userlog.py'
        )
    )


    ### transaction feature store
    bronze_transaction = BashOperator(
        task_id="bronze_transaction",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/bronze_transaction.py'
        )
    )
    silver_transaction = BashOperator(
        task_id="silver_transaction",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/silver_transaction.py'
        )    
    )


    ### gold feature store
    gold_feature_store = BashOperator(
        task_id="gold_feature_store",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/gold_feature_store.py'
        )
    )
    feature_store_completed = BashOperator(
        task_id="feature_store_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data/feature/feature_store_completed.py'
        )
    )

    ## ---- train/AutoML ----
    model_automl_start = DummyOperator(task_id="model_automl_start")

    model_1_automl = BashOperator(
    task_id="model_1_automl",
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 model/train_lr.py --date {{ ds }}'
        )
    )

    model_2_automl = BashOperator(
    task_id="model_2_automl",
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 model/train_rf.py --date {{ ds }}'
        )
    )

    model_automl_completed = DummyOperator(task_id="model_automl_completed")


    ## ---- inference ----
    model_inference_start = DummyOperator(task_id="model_inference_start")
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model/model_inference.py --date {{ ds }}'
        )
    )
    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    model_monitor_start = DummyOperator(task_id="model_monitor_start")
    model_monitor= BashOperator(
        task_id="model_monitor",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model/model_monitor.py --date {{ ds }}'
        )
    )
    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")


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

    ### train
    [label_store_completed, feature_store_completed] >> model_automl_start
    model_automl_start >> [model_1_automl, model_2_automl]
    [model_1_automl, model_2_automl] >> model_automl_completed

    ### inference
    feature_store_completed >> model_inference_start >> model_inference >> model_inference_completed >>model_monitor_start >>model_monitor >> model_monitor_completed
