import pandas as pd

# data injestion function
def download_fema_data(endpoint, columns, filename, is_large):
    base_url = f"https://www.fema.gov/api/open/{endpoint}"
    if is_large:
        url = f"{base_url}?$select={columns}&$format=csv&$allrecords=true&$filter=declarationDate%20gt%20'2000-01-01'"
    else:
        url = f"{base_url}?$select={columns}&$format=csv&$allrecords=true"
    df = pd.read_csv(url, low_memory=False)
    df.to_csv(f"{filename}.csv", index=False)
    print(f"Saved: data/{filename}.csv ({len(df)} rows)")


# Disaster Declarations
download_fema_data(
    "v2/DisasterDeclarationsSummaries", 
    "disasterNumber,state,incidentType,declarationDate,incidentBeginDate,incidentEndDate,declarationType",
    "disaster_declarations",
    False
)

# Public Assistance Funding
download_fema_data(
    "v2/PublicAssistanceFundedProjectsDetails", 
    "disasterNumber,incidentType,damageCategoryCode,stateNumberCode,gmProjectId,countyCode,projectAmount,federalShareObligated,totalObligated",
    "pa_funding_details",
    True
)
