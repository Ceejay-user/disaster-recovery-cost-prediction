import pandas as pd
import numpy as np
import us
import cpi
# cpi.update()
from datetime import datetime
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

decl_df, summaries_df, nri_df, pop_df = get_merged_data()


class DisasterDataProcessor:
    """
    Handles cleaning and feature engineering for FEMA disaster data.
    Implements domain-aware imputation for missing dates and early-stage 
    feature extraction for cost prediction.
    """

    def __init__(self, nri_df, pop_df):
        self.date_columns = ['declarationDate', 'incidentBeginDate', 'incidentEndDate']
        self.funding_components = ['totalObligatedAmountPa', 'totalAmountIhpApproved', 'totalObligatedAmountHmgp']
        self.nri_df = nri_df
        self.pop_df = pop_df
        # Placeholders for values 'learned' during training
        self.median_map = None
        self.top_types = None

    def _convert_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure all date columns are timezone-naive datetime objects."""
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
        return df

    def clean_and_merge(self, decl_df: pd.DataFrame, summaries_df: pd.DataFrame) -> pd.DataFrame:
        """
        TRAINING ONLY:
        Cleans data types and imputes missing incidentEndDates using median 
        durations per incident type.
        """
        decl_df = decl_df.copy()
        summaries_df = summaries_df.copy()
        nri_df = self._nri_df_prep(self.nri_df)
        pop_df = self._pop_df_prep(self.pop_df)

        decl_df = self._convert_to_datetime(decl_df)
        # LEARN: Calculate median durations and save to self.median_map
        temp_dur = (decl_df['incidentEndDate'] - decl_df['incidentBeginDate']).dt.days
        self.median_map = temp_dur.groupby(decl_df['incidentType']).median().to_dict()
        # Impute missing dates using the logic we just learned
        decl_df = self._impute_missing_dates(decl_df)
        # Map FEMA names to FIPS in disaster declarations
        decl_df = self._create_fips_crosswalk(decl_df)

        # 2. Join NRI (Static Risk/Assets)
        merged_df = decl_df.merge(nri_df, left_on='fips', right_on='STCOFIPS', how='left')

        # 3. Join Population (Temporal/Historical)
        # We round the fyDeclared down to the nearest decade to match your CSV columns
        merged_df['pop_decade'] = (merged_df['fyDeclared'] // 10) * 10
        merged_df = merged_df.merge(
            pop_df, 
            left_on=['fips', 'pop_decade'], 
            right_on=['cty_fips', 'year'], 
            how='left'
        )
        
        # Aggregate the merged table before merging with summaries table
        merged_df_agg = merged_df.groupby('disasterNumber').aggregate({
            'state': 'first',
            'declarationType': 'first',
            'fyDeclared': 'first',
            'designatedArea': 'nunique',
            'incidentType': 'first',
            'declarationDate': 'first',
            'incidentBeginDate': 'first',
            'incidentEndDate': 'first',
            'population': 'sum',      # Total people at risk in THAT decade
            'BUILDVALUE': 'sum',      # Infrastructure value
            'AGRIVALUE': 'sum',       # Agricultural value
            'RISK_SCORE': 'mean',
            'EAL_SCORE': 'mean',      # Historical financial pain
            'SOVI_SCORE': 'mean',     # Social vulnerability
            'ihProgramDeclared': 'max',
            'iaProgramDeclared': 'max',
            'paProgramDeclared': 'max',
            'hmProgramDeclared': 'max',
            'tribalRequest': 'max'
        }).reset_index()
        
        merged_df_agg['countiesAffected'] = merged_df_agg['designatedArea']
        full_df = summaries_df.merge(merged_df_agg, on='disasterNumber', how='left')

        # Fill Program Flags
        program_cols = ['iaProgramDeclared', 'paProgramDeclared', 'hmProgramDeclared', 'tribalRequest']
        full_df[program_cols] = full_df[program_cols].fillna(0).astype(int)
        
        return full_df
    
    def _impute_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to fill missing end dates using the SAVED median_map."""
        df = df.copy()
        mask = df['incidentEndDate'].isna()
        if mask.any() and self.median_map is not None:
            inferred_days = df.loc[mask, 'incidentType'].map(self.median_map).fillna(7) # 7 day default
            df.loc[mask, 'incidentEndDate'] = (
                df.loc[mask, 'incidentBeginDate'] + pd.to_timedelta(inferred_days, unit='D')
            )
        return df
    
    def _pop_df_prep(self, raw_pop_df: pd.DataFrame) -> pd.DataFrame:
        """Helper to prepare population df"""
        # Keep FIPS, drop county name (use NRI for names), melt the years
        raw_pop_df = raw_pop_df.copy()
        pop_long = raw_pop_df.melt(
            id_vars=['cty_fips'], 
            value_vars=[f'pop_{y}' for y in range(1900, 2030, 10)],
            var_name='year', 
            value_name='population'
        )
        # Clean 'pop_1990' -> 1990 (integer)
        pop_long['year'] = pop_long['year'].str.replace('pop_', '').astype(int)
        pop_long['cty_fips'] = pop_long['cty_fips'].astype(str).str.zfill(5)
        return pop_long
    
    def _nri_df_prep(self, nri_df: pd.DataFrame) -> pd.DataFrame:
        """Helper to prepare NRI df"""
        nri_df = nri_df.copy()
        nri_cols = ['STCOFIPS', 'BUILDVALUE', 'AGRIVALUE', 'RISK_SCORE', 'EAL_SCORE', 'SOVI_SCORE', 'RESL_SCORE']
        nri_df = nri_df[nri_cols]
        nri_df['STCOFIPS'] = nri_df['STCOFIPS'].astype(str).str.zfill(5)
        return nri_df
    
    def _create_fips_crosswalk(self, decl_df):
        """
        Helper: Constructs a 5-digit STCOFIPS from existing FEMA FIPS columns.
        Example: State '40' + County '153' -> '40153'
        """
        # 1. Ensure columns are strings and zero-padded
        # State FIPS must be 2 digits (e.g., '01')
        state_part = decl_df['fipsStateCode'].astype(str).str.zfill(2)
        
        # County FIPS must be 3 digits (e.g., '001')
        county_part = decl_df['fipsCountyCode'].astype(str).str.zfill(3)

        # 2. Combine them into the universal 5-digit key
        decl_df['fips'] = state_part + county_part

        # 3. Validation Check (Optional but Professional)
        # Most FEMA county FIPS are 5 digits. Some 'state-wide' records might be different.
        # We filter out records that don't have a valid county code (e.g., '000')
        decl_df = decl_df[decl_df['fipsCountyCode'].astype(int) > 0].reset_index(drop=True)
        return decl_df
    
    
    def engineer_early_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Creates features likely known shortly after declaration. """
        df = df.copy()
        df = self._convert_to_datetime(df)
        
        # 1. Temporal Features
        df['incident_duration_days'] = (df['incidentEndDate'] - df['incidentBeginDate']).dt.days
        df['incident_to_dec_lag'] = (df['declarationDate'] - df['incidentBeginDate']).dt.days
        df['month_declared'] = df['declarationDate'].dt.month

        # 2. Clip and Group
        num_cols = ['incident_duration_days', 'incident_to_dec_lag']
        df[num_cols] = df[num_cols].clip(lower=0)

        # Handle top types logic
        if self.top_types is None: # Only run during training
            counts = df['incidentType'].value_counts()
            self.top_types = counts[counts >= 20].index.tolist()
            
        df['incident_type_grouped'] = df['incidentType'].apply(
            lambda x: x if x in self.top_types else 'Other'
        )
        return df

    def engineer_target(self, df: pd.DataFrame, threshold: float = 2.8) -> pd.DataFrame:
        """
        Calculates the target 'total_recovery_cost' by summing funding components.
        Treats NaNs as 0 (no funds obligated).
        """
        df = df.copy()
        df[self.funding_components] = df[self.funding_components].fillna(0)
        df['total_recovery_cost'] = df[self.funding_components].sum(axis=1)
        # Drop rows where target cost is 0 (Addressing Zero-Inflation)
        df = df[df['total_recovery_cost'] > 0].reset_index(drop=True)

        year_col = 'fyDeclared' if 'fyDeclared' in df.columns else 'year'
        # Convert to integer and drop rows with missing values
        df = df.dropna(subset=[year_col, 'total_recovery_cost'])
        df[year_col] = df[year_col].clip(upper=2024).astype(int)

        # Calculate the inflation multiplier for each unique year ONCE
        unique_years = df[year_col].unique()
        
        # We find out how much $1 from each year is worth in 2024
        inflation_map = {
            yr: cpi.inflate(1, yr, to=2024) for yr in unique_years
        }
        # Apply the multiplier to the whole column at once (Vectorised)
        df['total_recovery_cost_adj'] = df['total_recovery_cost'] * df[year_col].map(inflation_map)

        # 2. Create a log-scaled version (for EDA only): NOT FOR MODEL
        df['log_total_recovery_cost_adj'] = np.log1p(df['total_recovery_cost_adj'])
        initial_count = len(df)
        # Calculate Z-score for removing outliers (to be observed in EDA): NOT FOR MODEL
        z_scores = (df['log_total_recovery_cost_adj'] - df['log_total_recovery_cost_adj'].mean()) / df['log_total_recovery_cost_adj'].std()
        # Apply the z score: keep only rows within the threshold (filter)
        df = df[z_scores.abs() <= threshold].reset_index(drop=True)

        # USE THE RAW 'total_recovery_cost_adj', NOW WITHOUT OUTLIERS FOR MODELING
        
        print(f"DEBUG: Threshold {threshold} | Before: {initial_count} | After: {len(df)}")
        return df
        
    
    def handle_feature_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Caps extreme feature values at the 99th percentile.
        This keeps the 'Severity' of the disaster but removes 'Impossible' noise.
        """
        df = df.copy()
        features_to_cap = ['incident_duration_days', 'incident_to_dec_lag', 'population', 'BUILDVALUE', 'AGRIVALUE', 'countiesAffected']
        
        for col in features_to_cap:
            upper_limit = df[col].quantile(0.99) # Top 1% threshold
            df[col] = df[col].clip(upper=upper_limit, lower=0)

            # Create Log versions of these features
            # This 'squashes' the outliers and makes the distribution more normal
            df[f'log_{col}'] = np.log1p(df[col])   
        return df

    
    def run_training_pipeline(self, decl_df: pd.DataFrame, summaries_df: pd.DataFrame) -> pd.DataFrame:
        """Use this when training the model."""
        df = self.clean_and_merge(decl_df, summaries_df)
        df = self.engineer_early_features(df)
        df = self.engineer_target(df)
        df = self.handle_feature_outliers(df)
        return df


    def run_inference_pipeline(self, df_inference: pd.DataFrame) -> pd.DataFrame:
        """Use this for your API. Takes only ONE dataframe."""
        df = df_inference.copy()
        df = self._convert_to_datetime(df)
        # Fill missing dates using the map learned in training
        df = self._impute_missing_dates(df)
        df = self.engineer_early_features(df)
        df = self.handle_feature_outliers(df)
        return df

if __name__ == "__main__":
    # Initialize the class instance
    processor = DisasterDataProcessor(nri_df, pop_df)
    
    processed_df = processor.run_training_pipeline(decl_df, summaries_df)
    
    print("Pipeline Success! Final columns:", processed_df.columns.tolist())
    print(processed_df.head())