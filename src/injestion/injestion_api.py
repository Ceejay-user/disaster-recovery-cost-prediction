import pandas as pd

# data injestion function
def download_fema_data(endpoint, columns, filename):
    base_url = f"https://www.fema.gov/api/open/{endpoint}"
    url = f"{base_url}?$select={columns}&$format=csv&$allrecords=true"
    df = pd.read_csv(url, low_memory=False)
    df.to_csv(f"../../data/raw/{filename}.csv", index=False)
    print(f"Saved: data/{filename}.csv ({len(df)} rows)")


# Disaster Declarations
download_fema_data(
    "v2/DisasterDeclarationsSummaries",
    "disasterNumber,state,designatedArea,incidentType,declarationDate,incidentBeginDate,incidentEndDate,declarationType,ihProgramDeclared,iaProgramDeclared,paProgramDeclared,hmProgramDeclared",
    "disaster_declarations"
)

# Public Assistance Funding
download_fema_data(
    "v2/PublicAssistanceFundedProjectsDetails", 
    "disasterNumber,projectAmount,federalShareObligated,totalObligated",
    "pa_funding_details"
)

# Disaster Summaries
download_fema_data(
    "v1/FemaWebDisasterSummaries",
    "disasterNumber,totalObligatedAmountPa,totalObligatedAmountCatAb,totalObligatedAmountCatC2g,totalAmountIhpApproved,totalObligatedAmountHmgp,totalNumberIaApproved",
    "disaster_summaries"
)

# Disaster Web Declarations
download_fema_data(
    "v1/FemaWebDisasterDeclarations",
    "disasterNumber,declarationType,incidentType,stateCode,region,declarationDate,declarationRequestDate,incidentBeginDate,incidentEndDate,iaProgramDeclared,paProgramDeclared,hmProgramDeclared",
    "web_disaster_declarations"
)