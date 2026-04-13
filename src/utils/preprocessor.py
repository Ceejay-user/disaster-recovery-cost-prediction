from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

def get_preprocessing_pipeline():
    """Defines the reusable professional ColumnTransformer."""
    
    # Feature Grouping
    num_log_features = ['log_incident_duration_days', 'log_incident_to_dec_lag', 'log_population', 'log_BUILDVALUE', 'log_AGRIVALUE', 'log_countiesAffected']
    # num_log_features = ['incident_duration_days', 'incident_to_dec_lag', 'population', 'BUILDVALUE', 'AGRIVALUE', 'countiesAffected']
    num_linear_features = ['fyDeclared', 'RISK_SCORE', 'SOVI_SCORE', 'EAL_SCORE']
    high_card_features = ['state']
    low_card_features = ['declarationType', 'incident_type_grouped']
    binary = ['iaProgramDeclared', 'paProgramDeclared', 'hmProgramDeclared', 'tribalRequest']

    # Build the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # Scaling numerical
            ('num', StandardScaler(), num_log_features + num_linear_features),
            
            # Target Encoding for High Cardinality
            ('target_enc', TargetEncoder(), high_card_features),
            
            # One-Hot for Low Cardinality
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_features),
            
            # Binary flags
            ('binary', 'passthrough', binary)
        ],
        remainder='drop'
    )
    return preprocessor