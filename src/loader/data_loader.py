from pathlib import Path
import pandas as pd

def get_merged_data(path_to_data=None):
    if path_to_data is None:
        # Check if __file__ exists (it won't in interactive/notebook modes)
        if "__file__" in globals():
            # Path relative to this script: src/loader/data_loader.py -> up 3 -> data/raw
            base_path = Path(__file__).resolve().parent.parent.parent
            path_to_data = base_path / "data"
        else:
            # Fallback for Notebooks/Interactive windows at the project root
            path_to_data = Path("data")

    decl_df = pd.read_csv(Path(path_to_data) / "raw" / "disaster_declarations.csv")
    summaries_df = pd.read_csv(Path(path_to_data) / "raw" / "disaster_summaries.csv")
    nri_df = pd.read_csv(Path(path_to_data) / "external" / "NRI_Table_Counties.csv")
    pop_df = pd.read_csv(Path(path_to_data) / "external" / "historical_county_populations.csv")
    # df_funding = pd.read_csv(Path(path_to_data) / "pa_funding_details.csv")

    
    return decl_df, summaries_df, nri_df, pop_df