from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .commons import DataFrameStructure

class PredictionRequest(BaseModel):
    """Schema for submitting new data for prediction."""
    data: List[Dict[str, Any]] = Field(..., description="List of records (rows) as dictionaries, where keys are feature names.")

class PredictionResponse(BaseModel):
    """Schema for prediction"""
    session_id: int = Field(..., description="The original AutoML session ID the model belongs to.")
    finalized_model_id: int = Field(..., description="The ID of the finalized model used for prediction.")
    preview_predictions: DataFrameStructure = Field(..., description="Preview of the first 10 rows (or fewer if less data) including prediction columns.")
    total_rows_processed: int = Field(..., description="The total number of rows processed from the uploaded CSV.")
    message: str = Field("Preview generated successfully. Full results are not returned by this endpoint.", description="Status message.")
    full_csv_base64: Optional[str] = Field(None, description="Base64 encoded string of the full prediction results CSV.")