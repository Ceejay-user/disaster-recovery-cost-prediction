import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import joblib
import os

# Import custom modules 
import sys
from pathlib import Path
# 1. Get the absolute path
current_file = Path(__file__).resolve()
# 2. Go up 2 levels
project_root = current_file.parent.parent.parent

# Add the project root to sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Modular imports
from src.loader.data_loader import get_merged_data
from src.preprocessing.features import DisasterDataProcessor
from src.utils.preprocessor import get_preprocessing_pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_optimized():
    # 1. Data Prep
    decl_df, summaries_df, nri_df, pop_df = get_merged_data()
    processor = DisasterDataProcessor(nri_df, pop_df)
    df_cleaned = processor.run_training_pipeline(decl_df, summaries_df)
    
    # We ensure year_declared and target are ready
    X = df_cleaned.drop(columns=['total_recovery_cost_adj'])
    y = df_cleaned['total_recovery_cost_adj']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define the Intelligent Mutual Info Function
    # We need the preprocessor to identify column names for the discrete mask
    col_transformer = get_preprocessing_pipeline()
    col_transformer.fit(X_train, y_train)

    def mi_score_func(X_transformed, y_log):
        """Identifies discrete features based on One-Hot/Binary prefixes."""
        names = col_transformer.get_feature_names_out()
        # True if name contains 'onehot' or 'binary', False if 'num' or 'target_enc'
        is_discrete = [('onehot' in n or 'binary' in n) for n in names]
        return mutual_info_regression(X_transformed, y_log, discrete_features=is_discrete, random_state=42)

    # setting mlflow experiment
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("FEMA_Disaster_Recovery_Cost_Model")

    with mlflow.start_run(run_name="RF_Optimized"):
        # 3. Build the Pipeline
        full_pipeline = Pipeline([
            ('preprocessor', col_transformer),
            ('feature_select', SelectKBest(score_func=mi_score_func)),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        wrapped_model = TransformedTargetRegressor(
            regressor=full_pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        # Parameter Grid for Wrapper -> Pipeline -> Steps
        param_dist = {
            'regressor__feature_select__k': [10, 15, 20, 'all'],
            'regressor__regressor__n_estimators': [100, 300, 500],
            'regressor__regressor__max_depth': [10, 20, None],
            'regressor__regressor__min_samples_leaf': [1, 2, 4, 5],
            'regressor__regressor__max_features': ['sqrt', 'log2', None],
            'regressor__regressor__max_samples': [0.5]
        }

        # 5. Execute Search (n_iter=30 for thorough exploration)
        search = RandomizedSearchCV(
            wrapped_model, 
            param_distributions=param_dist, 
            n_iter=30, 
            cv=5, 
            scoring='neg_mean_absolute_error', 
            n_jobs=-1, 
            random_state=42,
            verbose=1
        )
        
        print("Starting optimization (n_iter=30)...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # 4. Logging to MLflow
        mlflow.log_params(search.best_params_)

        # Log Metrics
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({
            "mae_dollars": mae,
            "rmse_dollars": rmse,
            "r2_score": r2
        })

        # 6. Save and Log the Stateful processor
        # 'processor' is the instance you used to clean X_train (before search.fit)
        processor_filename = "processor.joblib"
        joblib.dump(processor, processor_filename)

        # Log the local file as an artifact in MLflow
        mlflow.log_artifact(processor_filename)

        # Clean up the local file after logging
        if os.path.exists(processor_filename):
            os.remove(processor_filename)

        # Log the Model
        mlflow.sklearn.log_model(sk_model=best_model, name="model")
        
        # Log the features selected by mutual info
        # 1. Get the preprocessor and selector from the best model
        preprocessor = best_model.regressor_.named_steps['preprocessor']
        selector = best_model.regressor_.named_steps['feature_select']

        # 2. Get all names after preprocessor, then filter by selector support
        all_feature_names = preprocessor.get_feature_names_out()
        selected_features = all_feature_names[selector.get_support()].tolist()

        # 3. Log as a parameter (useful for quick viewing in UI)
        # We join with a comma because MLflow parameters have a character limit
        mlflow.log_param("selected_features_list", ", ".join(selected_features))
        
        # 4. Log as a text artifact (best for long lists / documentation)
        with open("selected_features.txt", "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        mlflow.log_artifact("selected_features.txt")

        print(f"Logged {len(selected_features)} features to MLflow.")


        # 7. SHAP Explainability
        # Transform X_test and filter names based on what SelectKBest chose
        X_test_transformed = preprocessor.transform(X_test)
        selected_names = preprocessor.get_feature_names_out()[selector.get_support()]
        X_test_selected = X_test_transformed[:, selector.get_support()]

        # Generate SHAP values for the internal RF
        explainer = shap.TreeExplainer(best_model.regressor_.named_steps['regressor'])
        shap_values_obj = explainer(X_test_selected)
        shap_values_obj.feature_names = list(selected_names)

        # Log Summary (Beeswarm)
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values_obj, show=False)
        plt.savefig("shap_summary.png", bbox_inches='tight')
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        # Log Waterfall (First prediction)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_obj[0], show=False)
        plt.savefig("shap_waterfall.png", bbox_inches='tight')
        mlflow.log_artifact("shap_waterfall.png")
        plt.close()

        print(f"Optimization Complete. Best MAE: ${mae:,.2f} | R2: {r2:.4f}")

if __name__ == "__main__":
    train_optimized()
