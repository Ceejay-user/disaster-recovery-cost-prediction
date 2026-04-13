from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import date
from contextlib import asynccontextmanager


# Configurations
TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fema_recovery_model"
MODEL_VERSION = 2
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        # Load the model (TransformedTargetRegressor wrapper)
        model = mlflow.pyfunc.load_model(MODEL_URI)
        ml_models['pipeline'] = model

        # Get run ID
        run_id = model.metadata.run_id

        # Download the Trained processor artifact
        # This downloads 'processor.joblib' from the MLflow run to a local folder
        local_path = client.download_artifacts(run_id, "processor.joblib", dst_path=".")
        ml_models['processor'] = joblib.load(local_path)
        print("MLflow Model and Processor loaded successfully")
    except Exception as e:
        print(f"Error during lifespan setup: {e}")
    yield
    ml_models.clear()

app = FastAPI(
    title='FEMA Disaster Cost Predictor',
    version='1.0.0',
    lifespan=lifespan
)

# User-friendly schema (No logs or complex math required from user)
class PredictRequest(BaseModel):
    state: str = Field(..., example="TX")
    declarationType: Literal['Major Disaster', 'Emergency', 'Fire Management']
    incidentType: str = Field(..., example="Flood")
    fyDeclared: int = Field(2023, ge=1900)
    # Raw dates for the backend to process
    declarationDate: date = Field(..., example="2023-10-01")
    declarationRequestDate: date = Field(..., example="2023-09-25")
    incidentBeginDate: date = Field(..., example="2023-09-20")
    incidentEndDate: date = Field(..., example="2023-09-30")
    # Program flags
    iaProgramDeclared: int = Field(0, ge=0, le=1)
    paProgramDeclared: int = Field(1, ge=0, le=1)
    hmProgramDeclared: int = Field(0, ge=0, le=1)
    tribalRequest: int = Field(0, ge=0, le=1)
    countiesAffected: int = Field(1, gt=0)

class PredictResponse(BaseModel):
    estimated_cost_usd: float
    model_version: str

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    model = ml_models.get('pipeline')
    processor = ml_models.get('processor')
    
    try:
        # 1. Capture the single record
        input_df = pd.DataFrame([request.model_dump()])

        # 3. Call the inference processor method
        processed_df = processor.run_inference_pipeline(input_df)

        # 4. Predict
        prediction = model.predict(processed_df)[0]
        
        return PredictResponse(
            estimated_cost_usd=round(float(prediction), 2),
            model_version=f"{MODEL_NAME} v{MODEL_VERSION}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
