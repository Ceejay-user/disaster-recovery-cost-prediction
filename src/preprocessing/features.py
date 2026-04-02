import pandas as pd
import numpy as np
from typing import List, Optional

import sys
from pathlib import Path

# 1. Get the absolute path of features.py
# 2. Go up 2 levels (from src/preprocessing/ to the project root)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

# Add the project root to sys.path so 'from src.loader...' works
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.loader.data_loader import get_merged_data

class DisasterDataProcessor:
    """
    Handles cleaning and feature engineering for FEMA disaster data.
    Implements domain-aware imputation for missing dates and early-stage 
    feature extraction for cost prediction.
    """

    def __init__(self):
        self.date_columns = [
            'declarationDate', 'declarationRequestDate', 
            'incidentBeginDate', 'incidentEndDate'
        ]
        self.funding_components = [
            'totalObligatedAmountPa', 
            'totalAmountIhpApproved', 
            'totalObligatedAmountHmgp'
        ]

    def _convert_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure all date columns are timezone-naive datetime objects."""
        for col in self.date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
        return df

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data types and imputes missing incidentEndDates using median 
        durations per incident type.
        
        Args:
            df (pd.DataFrame): The merged raw DataFrame.
            
        Returns:
            pd.DataFrame: Cleaned and imputed DataFrame.
        """
        df = df.copy()
        df = self._convert_to_datetime(df)

        # 1. Impute incidentEndDate using your logic
        # Calculate existing durations for the mapping
        temp_dur = (df['incidentEndDate'] - df['incidentBeginDate']).dt.days
        median_map = temp_dur.groupby(df['incidentType']).median()
        
        # Apply the mapping to missing values
        mask = df['incidentEndDate'].isna()
        inferred_days = df.loc[mask, 'incidentType'].map(median_map)
        
        # Fill missing dates: BeginDate + Median Duration
        df.loc[mask, 'incidentEndDate'] = (
            df.loc[mask, 'incidentBeginDate'] + 
            pd.to_timedelta(inferred_days, unit='D')
        )

        # 2. Fill Program Flags (Categorical/Boolean flags)
        program_cols = ['iaProgramDeclared', 'paProgramDeclared', 'hmProgramDeclared']
        df[program_cols] = df[program_cols].fillna(0).astype(int)
        
        # 3. Fill missing declarationRequestDate with declarationDate (Best proxy)
        df['declarationRequestDate'] = df['declarationRequestDate'].fillna(df['declarationDate'])

        return df

    def engineer_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the target 'total_recovery_cost' by summing funding components.
        Treats NaNs as 0 (no funds obligated).
        """
        df = df.copy()
        df[self.funding_components] = df[self.funding_components].fillna(0)
        df['total_recovery_cost'] = df[self.funding_components].sum(axis=1)
        return df

    def engineer_early_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features likely known shortly after declaration.
        
        Args:
            df (pd.DataFrame): The cleaned DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.copy()

        # 1. Temporal Features (Known at declaration)
        df['incident_duration_days'] = (df['incidentEndDate'] - df['incidentBeginDate']).dt.days
        df['request_to_dec_lag'] = (df['declarationDate'] - df['declarationRequestDate']).dt.days
        df['incident_to_dec_lag'] = (df['declarationDate'] - df['incidentBeginDate']).dt.days
        
        # 2. Seasonality
        df['month_declared'] = df['declarationDate'].dt.month

        # 3. Handling negative durations (FEMA data can have errors)
        # Professional standard: Clip at 0 to avoid model confusion
        num_cols = ['incident_duration_days', 'request_to_dec_lag', 'incident_to_dec_lag']
        df[num_cols] = df[num_cols].clip(lower=0)

        # 3. Categorical Consolidation (Optional but professional)
        # Low frequency incident types can be grouped as 'Other' to prevent overfitting
        top_types = df['incidentType'].value_counts().nlargest(10).index
        df['incident_type_grouped'] = df['incidentType'].apply(lambda x: x if x in top_types else 'Other')

        # 4. Regional Scaling
        # totalNumberIaApproved is known early for IA programs
        df['totalNumberIaApproved'] = df['totalNumberIaApproved'].fillna(0)

        return df
    
    def filter_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes zero-cost rows and drops redundant/raw columns.
        """
        df = df.copy()

        # 1. Drop rows where target cost is 0 (Addressing Zero-Inflation)
        df = df[df['total_recovery_cost'] > 0].reset_index(drop=True)

        # 2. List of columns to drop (Raw dates, helper columns, and specific components)
        cols_to_drop = [
            'declarationDate', 'declarationRequestDate', 
            'incidentBeginDate', 'incidentEndDate',
            'totalObligatedAmountPa', 'totalAmountIhpApproved', 'totalObligatedAmountHmgp',
            'totalNumberIaApproved', 'incidentType'
        ]
        
        # Adding the specific columns you mentioned
        extra_drops = ['totalObligated', 'totalObligatedAmountCatAb', 'totalObligatedAmountCatC2g']
        
        # Combine and drop only if they exist in the dataframe
        final_drops = [c for c in (cols_to_drop + extra_drops) if c in df.columns]
        df = df.drop(columns=final_drops)

        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline execution."""
        df = self.clean_and_impute(df)
        df = self.engineer_target(df)
        df = self.engineer_early_features(df)
        df = self.filter_and_select(df)
        return df

# if __name__ == "__main__":
#     # Initialize the class instance
#     processor = DisasterDataProcessor()
    
#     # Get data and run
#     raw_data = get_merged_data()
#     processed_df = processor.run_pipeline(raw_data)
    
#     print("Pipeline Success! Final columns:", processed_df.columns.tolist())
#     print(processed_df.head())
