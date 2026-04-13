import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Import custom modules
# 1. Get the absolute path of features.py
# 2. Go up 2 levels (from src/preprocessing/ to the project root)
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

# Add the project root to sys.path so 'from src.loader...' works
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.loader.data_loader import get_merged_data
from src.utils.preprocessor import get_preprocessing_pipeline
from src.preprocessing.features import DisasterDataProcessor

# Import Base Models for Comparison
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor



def log_feature_importance(model, name):
    """
    Extracts importance, prints to console, and logs a plot + CSV to MLflow.
    """
    pipeline = model.regressor_
    rf_regressor = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Reconstruct names exactly as they exited the ColumnTransformer
    num_log = ['log_incident_duration_days', 'log_incident_to_dec_lag', 'log_population', 'log_BUILDVALUE', 'log_AGRIVALUE', 'log_countiesAffected']
    num_linear = ['fyDeclared', 'RISK_SCORE', 'SOVI_SCORE', 'EAL_SCORE']
    high_card_names = ['state']
    ohe_names = preprocessor.named_transformers_['onehot'].get_feature_names_out()
    binary = ['iaProgramDeclared', 'paProgramDeclared', 'hmProgramDeclared', 'tribalRequest']

    all_names = np.concatenate([num_log, num_linear, high_card_names, ohe_names, binary])

    feat_imp = pd.DataFrame({
        'feature': all_names,
        'importance': rf_regressor.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # 1. Log CSV as an artifact
    feat_imp.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # 2. Create and log a Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp.head(15), x='importance', y='feature')
    plt.title(f"Top 15 Drivers - {name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    print("\n--- Top 10 Recovery Cost Drivers ---")
    print(feat_imp.head(10).to_string(index=False))

def run_benchmarking():
    # 1. Data Prep
    decl_df, summaries_df, nri_df, pop_df = get_merged_data()
    processor = DisasterDataProcessor(nri_df, pop_df)
    df = processor.run_training_pipeline(decl_df, summaries_df)
    df = df.drop(columns=['totalObligatedAmountCatAb', 'totalObligatedAmountCatC2g', 'totalNumberIaApproved'])
    df = df.dropna()

    X = df.drop(columns=['total_recovery_cost_adj'])
    y = df['total_recovery_cost_adj']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Initialize MLflow Experiment
    # mlflow.set_tracking_uri('http://127.0.0.1:5000')
    # mlflow.set_experiment("FEMA_Disaster_Recovery_Cost_Model")

    base_models = {
        "Base Linear Regression": LinearRegression(),
        "Base Ridge Regression": Ridge(alpha=1.0),
        "Base Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Base K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        'Base Gradient Boosting': GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42,
        objective='reg:squarederror' 
    )
    }

    preprocessor = get_preprocessing_pipeline()

    for name, reg in base_models.items():

        ###########
        full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', reg)
            ])

        model = TransformedTargetRegressor(
            regressor=full_pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        # 3. Fit & Predict
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # 4. Calculate Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"\n{name}")
        print({
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })
        
        # Log specific details for tree-based models
        if name == "Base Random Forest":
            log_feature_importance(model, name)
        ###########

    pipeline = model.regressor_
    preprocessor = pipeline.named_steps['preprocessor']
    print(preprocessor.get_feature_names_out())

        # Start a unique MLflow run for each model
        # with mlflow.start_run(run_name=name):
            
        #     full_pipeline = Pipeline(steps=[
        #         ('preprocessor', preprocessor),
        #         ('regressor', reg)
        #     ])

        #     model = TransformedTargetRegressor(
        #         regressor=full_pipeline,
        #         func=np.log1p,
        #         inverse_func=np.expm1
        #     )

        #     # 3. Fit & Predict
        #     model.fit(X_train, y_train)
        #     preds = model.predict(X_test)

        #     # 4. Calculate Metrics
        #     mae = mean_absolute_error(y_test, preds)
        #     rmse = np.sqrt(mean_squared_error(y_test, preds))
        #     r2 = r2_score(y_test, preds)

        #     # 5. Log to MLflow
        #     mlflow.log_param("model_name", name)
        #     mlflow.log_metrics({
        #         "mae": mae,
        #         "rmse": rmse,
        #         "r2": r2
        #     })
            
        #     # Log specific details for tree-based models
        #     if name == "Base Random Forest":
        #         log_feature_importance(model, name)
        #         # Log the actual model file
        #         mlflow.sklearn.log_model(model, "model")

        #     print(f"\n--- {name} ---")
        #     print(f"R²: {r2:.4f} | MAE: ${mae:,.2f}")

if __name__ == "__main__":
    run_benchmarking()
