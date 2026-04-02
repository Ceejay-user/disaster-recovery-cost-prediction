from pathlib import Path
import pandas as pd

def get_merged_data(path_to_data=None):
    if path_to_data is None:
        # Check if __file__ exists (it won't in interactive/notebook modes)
        if "__file__" in globals():
            # Path relative to this script: src/loader/data_loader.py -> up 3 -> data/raw
            base_path = Path(__file__).resolve().parent.parent.parent
            path_to_data = base_path / "data" / "raw"
        else:
            # Fallback for Notebooks/Interactive windows at the project root
            path_to_data = Path("data/raw")

    df_decl = pd.read_csv(Path(path_to_data) / "web_disaster_declarations.csv")
    df_summaries = pd.read_csv(Path(path_to_data) / "disaster_summaries.csv")
    df_funding = pd.read_csv(Path(path_to_data) / "pa_funding_details.csv")

    # Aggregate funding
    df_funding_agg = df_funding.groupby('disasterNumber').aggregate({
        'totalObligated': 'sum'
    }).reset_index()

    # Drop duplicates
    df_funding_agg = df_funding_agg.drop_duplicates()

    # Merge
    merged = df_summaries.merge(df_decl, on='disasterNumber', how='left')
    merged = merged.merge(df_funding_agg, on='disasterNumber', how='left')
    
    return merged