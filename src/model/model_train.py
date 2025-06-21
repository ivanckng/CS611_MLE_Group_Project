
import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, fbeta_score, precision_score,
    recall_score, confusion_matrix, make_scorer
)

# Hyperparameter Optimization
try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Using default parameters.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnPredictionTrainer:

    def __init__(self,
                 bucket_name="cs611_mle",
                 start_date_str='2015-01-01',
                 end_date_str='2017-03-01',
                 random_state=11):
        self.bucket_name = bucket_name
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.random_state = random_state

        try:
            from google.cloud import storage
            import gcsfs
            self.storage_client = storage.Client()
            self.fs = gcsfs.GCSFileSystem()
        except ImportError:
            logger.warning("GCS libraries not available. Model saving may be limited.")
            self.storage_client = None
            self.fs = None

        # Model artifacts
        self.scaler = None
        self.best_model = None
        self.best_params = None
        self.best_threshold = None
        self.feature_columns = None

        # Results storage
        self.results = {}

    def generate_date_list(self):
        start_date = datetime.strptime(self.start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date_str, "%Y-%m-%d")

        date_list = []
        current_date = datetime(start_date.year, start_date.month, 1)

        while current_date <= end_date:
            current_year = current_date.year
            current_month = current_date.month
            if current_month < 10:
                date_str = f"{current_year}_0{current_month}_01"
            else:
                date_str = f"{current_year}_{current_month}_01"

            date_list.append(date_str)

            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        logger.info(f"Generated {len(date_list)} date periods for feature loading")
        return date_list

    def load_labels(self):
        logger.info("Loading labels data from GCS...")

        label_path_in_bucket = "Gold Layer/labels.csv"
        label_gcs_path = f"gs://{self.bucket_name}/{label_path_in_bucket}"

        try:
            label_df = pd.read_csv(label_gcs_path)
            logger.info(f"Loaded labels data from {label_gcs_path}: {label_df.shape[0]} rows")

            # Data preprocessing
            label_df = label_df.drop_duplicates(
                subset=['msno', 'membership_start_date', 'membership_expire_date'],
                keep='first'
            )

            # Convert dates
            label_df['membership_expire_date'] = pd.to_datetime(
                label_df['membership_expire_date'], format='%Y-%m-%d'
            )
            label_df['membership_start_date'] = pd.to_datetime(
                label_df['membership_start_date'], format='%Y-%m-%d'
            )

            # Calculate plan days and filter for 30/31 day plans
            label_df['plan_days'] = (
                    label_df['membership_expire_date'] - label_df['membership_start_date']
            )

            # Filter for 30 and 31 day plans only
            target_plan_days = [pd.Timedelta(days=30), pd.Timedelta(days=31)]
            label_df = label_df[label_df['plan_days'].isin(target_plan_days)]

            logger.info(f"After filtering for 30/31 day plans: {label_df.shape[0]} rows")
            return label_df

        except Exception as e:
            logger.error(f"Error loading labels: {str(e)}")
            raise

    def load_features(self):

        logger.info("Loading feature data from GCS...")

        dates_list = self.generate_date_list()
        feature_dataframes = []

        for date_str in dates_list:
            feature_path_in_bucket = f"datamart/gold/feature_store/gold_featurestore_{date_str}.parquet/"
            feature_gcs_path = f"gs://{self.bucket_name}/{feature_path_in_bucket}"

            try:
                df = pd.read_parquet(feature_gcs_path)
                feature_dataframes.append(df)
                logger.info(f"Loaded features for {date_str}: {df.shape[0]} rows")
            except Exception as e:
                logger.warning(f"Could not load features for {date_str}: {str(e)}")
                continue

        if not feature_dataframes:
            raise ValueError("No feature data could be loaded")

        # Combine all feature dataframes
        feature_df = pd.concat(feature_dataframes, ignore_index=True)
        logger.info(f"Combined feature data: {feature_df.shape[0]} rows")

        # Remove duplicates
        feature_df = feature_df.drop_duplicates(
            subset=['msno', 'membership_start_date', 'membership_expire_date'],
            keep='first'
        )

        # Convert dates
        feature_df['membership_start_date'] = pd.to_datetime(feature_df['membership_start_date'])
        feature_df['membership_expire_date'] = pd.to_datetime(feature_df['membership_expire_date'])

        # Calculate and filter plan days
        feature_df['plan_days'] = (
                feature_df['membership_expire_date'] - feature_df['membership_start_date']
        )

        # 30 and 31 day plans
        target_plan_days = [pd.Timedelta(days=30), pd.Timedelta(days=31)]
        feature_df = feature_df[feature_df['plan_days'].isin(target_plan_days)]

        logger.info(f"After processing: {feature_df.shape[0]} rows")
        return feature_df

    def create_final_dataset(self, label_df, feature_df):
        """Merge labels and features to create final modeling dataset"""
        logger.info("Creating final modeling dataset...")

        # Merge labels and features
        final_df = pd.merge(
            label_df,
            feature_df,
            how='left',
            on=['msno', 'membership_start_date', 'membership_expire_date']
        )

        # Remove rows with missing features
        final_df = final_df.dropna()
        final_df = final_df.reset_index(drop=True)

        # Drop unnecessary columns
        columns_to_drop = [
            'plan_days_x', 'plan_days_y', 'transaction_date'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in final_df.columns]
        final_df = final_df.drop(columns=columns_to_drop)

        # Convert categorical columns
        categorical_cols = [
            'registered_via', 'churn', 'payment_method_id',
            'is_auto_renew', 'is_cancel', 'city', 'gender'
        ]

        for col in categorical_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype('category')

        logger.info(f"Final dataset shape: {final_df.shape}")
        return final_df

    def split_data(self, final_df):
        """Split data into train, test, and OOT sets"""
        logger.info("Splitting data into train/test/OOT sets...")

        # Split by dates
        oot_df = final_df[
            (final_df['membership_start_date'] >= '2016-11-01') &
            (final_df['membership_start_date'] <= '2016-11-30')
            ]

        train_val_df = final_df[final_df['membership_start_date'] < '2016-11-01']

        # Split train_val into train and test
        train_df, test_df = train_test_split(
            train_val_df,
            test_size=0.2,
            shuffle=True,
            random_state=self.random_state,
            stratify=train_val_df['churn']
        )

        logger.info(f"Train set: {train_df.shape[0]} rows")
        logger.info(f"Test set: {test_df.shape[0]} rows")
        logger.info(f"OOT set: {oot_df.shape[0]} rows")

        return train_df, test_df, oot_df

    def preprocess_features(self, train_df, test_df, oot_df):
        logger.info("Preprocessing features...")

        # Define feature columns
        categorical_cols = ['payment_method_id', 'is_auto_renew', 'is_cancel',
                            'city', 'gender', 'registered_via']

        exclude_cols = ['msno', 'membership_start_date', 'membership_expire_date', 'churn']

        # Separate features and targets
        X_train = train_df.drop(columns=exclude_cols)
        X_test = test_df.drop(columns=exclude_cols)
        X_oot = oot_df.drop(columns=exclude_cols)

        y_train = train_df['churn']
        y_test = test_df['churn']
        y_oot = oot_df['churn']

        # Process categorical variables
        X_train, X_test, X_oot = self._process_categorical_features(
            X_train, X_test, X_oot, categorical_cols
        )

        # Separate numeric and categorical features
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

        X_train_numeric = X_train[numeric_cols]
        X_test_numeric = X_test[numeric_cols]
        X_oot_numeric = X_oot[numeric_cols]

        X_train_cat = X_train[categorical_cols]
        X_test_cat = X_test[categorical_cols]
        X_oot_cat = X_oot[categorical_cols]

        # Scale numeric features
        self.scaler = StandardScaler()
        X_train_numeric_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_numeric),
            columns=X_train_numeric.columns,
            index=X_train_numeric.index
        )

        X_test_numeric_scaled = pd.DataFrame(
            self.scaler.transform(X_test_numeric),
            columns=X_test_numeric.columns,
            index=X_test_numeric.index
        )

        X_oot_numeric_scaled = pd.DataFrame(
            self.scaler.transform(X_oot_numeric),
            columns=X_oot_numeric.columns,
            index=X_oot_numeric.index
        )

        # One-hot encode categorical features
        X_train_cat_encoded = pd.get_dummies(
            X_train_cat,
            columns=['payment_method_id', 'city', 'gender', 'registered_via'],
            dtype=int
        )
        X_test_cat_encoded = pd.get_dummies(
            X_test_cat,
            columns=['payment_method_id', 'city', 'gender', 'registered_via'],
            dtype=int
        )
        X_oot_cat_encoded = pd.get_dummies(
            X_oot_cat,
            columns=['payment_method_id', 'city', 'gender', 'registered_via'],
            dtype=int
        )

        # Combine features
        X_train_final = pd.concat([X_train_numeric_scaled, X_train_cat_encoded], axis=1)
        X_test_final = pd.concat([X_test_numeric_scaled, X_test_cat_encoded], axis=1)
        X_oot_final = pd.concat([X_oot_numeric_scaled, X_oot_cat_encoded], axis=1)

        # Apply get_dummies with drop_first=True
        X_train_final = pd.get_dummies(X_train_final, drop_first=True)
        X_test_final = pd.get_dummies(X_test_final, drop_first=True)
        X_oot_final = pd.get_dummies(X_oot_final, drop_first=True)

        # Store feature columns for later use
        self.feature_columns = X_train_final.columns.tolist()

        # Convert y to numeric
        y_train_numeric = y_train.cat.codes.values if hasattr(y_train, 'cat') else y_train.values
        y_train_numeric = np.where(y_train_numeric == -1, 0, y_train_numeric).astype(int)

        y_test_numeric = y_test.astype(int)
        y_oot_numeric = y_oot.astype(int)

        logger.info(f"Final feature shape: {X_train_final.shape}")

        return (X_train_final, X_test_final, X_oot_final,
                y_train_numeric, y_test_numeric, y_oot_numeric)

    def _process_categorical_features(self, X_train, X_test, X_oot, categorical_cols):
        """Process categorical features with grouping of rare categories"""

        # Payment method grouping
        if 'payment_method_id' in categorical_cols:
            payment_methods = ['41', '40', '36', '39', '37']

            def method_mapping(col):
                return col if str(col) in payment_methods else '99'

            X_train['payment_method_id'] = X_train['payment_method_id'].apply(method_mapping)
            X_test['payment_method_id'] = X_test['payment_method_id'].apply(method_mapping)
            X_oot['payment_method_id'] = X_oot['payment_method_id'].apply(method_mapping)

        # City grouping
        if 'city' in categorical_cols:
            cities = ['1', '13', '5', '4', '15', '22']

            def city_mapping(col):
                return col if str(col) in cities else '99'

            X_train['city'] = X_train['city'].apply(city_mapping)
            X_test['city'] = X_test['city'].apply(city_mapping)
            X_oot['city'] = X_oot['city'].apply(city_mapping)

        # Registered via grouping
        if 'registered_via' in categorical_cols:
            regist = ['7', '9', '3']

            def regist_mapping(col):
                return col if str(col) in regist else '99'

            X_train['registered_via'] = X_train['registered_via'].apply(regist_mapping)
            X_test['registered_via'] = X_test['registered_via'].apply(regist_mapping)
            X_oot['registered_via'] = X_oot['registered_via'].apply(regist_mapping)

        return X_train, X_test, X_oot

    def optimize_logistic_regression(self, X_train, y_train):
        """Optimize Logistic Regression using Optuna"""
        logger.info("Optimizing Logistic Regression hyperparameters...")

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear'
            )
            model.fit(X_train, y_train)

            # Calculate CV score manually
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')

            return model, {'C': 1.0, 'penalty': 'l2'}, cv_scores.mean()

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            try:
                params = {
                    'C': trial.suggest_float('C', 0.1, 50, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
                }

                model = LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    solver='liblinear',
                    **params
                )

                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=3, scoring='roc_auc', n_jobs=-1
                )

                return scores.mean()

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(objective, n_trials=20, show_progress_bar=False)

        # Train best model
        best_model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='liblinear',
            **study.best_params
        )

        best_model.fit(X_train, y_train)

        logger.info(f"Best LR params: {study.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")

        return best_model, study.best_params, study.best_value

    def find_optimal_threshold(self, model, X_train, y_train):
        # Find optimal threshold based on Recall
        logger.info("Finding optimal threshold...")

        train_pred_proba = model.predict_proba(X_train)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.01)

        results = []
        for thresh in thresholds:
            preds = (train_pred_proba >= thresh).astype(int)
            f1_5 = fbeta_score(y_train, preds, beta=1.5)
            prec = precision_score(y_train, preds)
            rec = recall_score(y_train, preds)
            tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
            results.append((thresh, f1_5, prec, rec, tp, fp, fn, tn))

        df_thresh = pd.DataFrame(results, columns=[
            "Threshold", "F1.5", "Precision", "Recall", "TP", "FP", "FN", "TN"
        ])

        best_threshold = df_thresh.loc[df_thresh["Recall"].idxmax(), "Threshold"]
        logger.info(f"Optimal threshold: {best_threshold:.3f}")

        return best_threshold

    def evaluate_model(self, model, X, y, threshold, dataset_name):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        auc = roc_auc_score(y, y_proba)
        f1_5 = fbeta_score(y, y_pred, beta=1.5)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        results = {
            'dataset': dataset_name,
            'auc': auc,
            'f1_5': f1_5,
            'precision': precision,
            'recall': recall,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'threshold': threshold,
            'total_users': len(y),
            'total_churn': int(y.sum()),
            'targeted': int((y_pred == 1).sum())
        }

        logger.info(f"{dataset_name} Results:")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  F1.5: {f1_5:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Precision: {precision:.4f}")

        return results

    def save_to_gcs(self, final_df, model_results):

        logger.info("Saving results to GCS...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save model artifacts using pandas-compatible approach
            if self.fs is not None:
                # Save model (joblib) - using gcsfs for binary files
                model_dir_in_bucket = f"models/"
                model_path = f"{model_dir_gcs_path}/lr_best_model.joblib"
                with self.fs.open(model_path, 'wb') as f:
                    joblib.dump(self.best_model, f)

                # Save model metadata
                metadata = {
                    'model_type': 'LogisticRegression',
                    'best_params': self.best_params,
                    'threshold': self.best_threshold,
                    'feature_columns': self.feature_columns,
                    'results': model_results,
                    'timestamp': timestamp,
                }

                metadata_path = f"{model_dir_gcs_path}/info.pkl"
                with self.fs.open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)

                # Save results summary
                results_path = f"{model_dir_gcs_path}/results_summary.pkl"
                with self.fs.open(results_path, 'wb') as f:
                    pickle.dump(model_results, f)

                logger.info(f"Saved model artifacts to: {model_dir_gcs_path}")
            else:
                logger.warning("GCS filesystem not available. Model artifacts not saved.")
                model_dir_gcs_path = None

            return {
                'gold_table_path': gold_table_gcs_path,
                'model_dir': model_dir_gcs_path,
                'timestamp': timestamp
            }

        except Exception as e:
            logger.error(f"Error saving to GCS: {str(e)}")
            raise

    def run_training_pipeline(self):
        logger.info("Starting churn prediction training pipeline...")

        try:
            # 1. Load data
            label_df = self.load_labels()
            feature_df = self.load_features()

            # 2. Create final dataset
            final_df = self.create_final_dataset(label_df, feature_df)

            # 3. Split data
            train_df, test_df, oot_df = self.split_data(final_df)

            # 4. Preprocess features
            (X_train, X_test, X_oot,
             y_train, y_test, y_oot) = self.preprocess_features(train_df, test_df, oot_df)

            # 5. Train model
            self.best_model, self.best_params, cv_score = self.optimize_logistic_regression(
                X_train, y_train
            )

            # 6. Find optimal threshold
            self.best_threshold = self.find_optimal_threshold(
                self.best_model, X_train, y_train
            )

            # 7. Evaluate model
            train_results = self.evaluate_model(
                self.best_model, X_train, y_train, self.best_threshold, "Train"
            )
            test_results = self.evaluate_model(
                self.best_model, X_test, y_test, self.best_threshold, "Test"
            )

            # 8. Compile results
            model_results = {
                'cv_score': cv_score,
                'best_params': self.best_params,
                'optimal_threshold': self.best_threshold,
                'train_results': train_results,
                'test_results': test_results,
            }

            # 9. Save to GCS
            save_info = self.save_to_gcs(final_df, model_results)

            logger.info("Training pipeline completed successfully!")
            logger.info(f"Test AUC: {test_results['auc']:.4f}")
            logger.info(f"Test F1.5: {test_results['f1_5']:.4f}")
            logger.info(f"Test Recall: {test_results['recall']:.4f}")

            return {
                'status': 'success',
                'results': model_results,
                'save_info': save_info
            }

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }


def main():
    try:
        # Initialize trainer
        trainer = ChurnPredictionTrainer()

        # Run training pipeline
        results = trainer.run_training_pipeline()

        if results['status'] == 'success':
            logger.info("Training pipeline completed successfully")
            return 0
        else:
            logger.error("Training pipeline failed")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
