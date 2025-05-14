from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.schemas.automl_jobs import AutoMLJobResultBase

class ExplainPredictionRequest(BaseModel):
    finalized_model_id: int = Field(...)
    data_instance: Dict[str, Any] = Field(...)

class ExplainResult(AutoMLJobResultBase):
    base_value: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None
    prediction_input: Optional[Dict[str, Any]] = None