from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.schemas.automl_jobs import AutoMLJobResultBase

class CheckDriftRequest(BaseModel):
    new_data_path: str = Field(...)
    report_save_path: Optional[str] = None

class DriftResult(AutoMLJobResultBase):
    drift_detected: Optional[bool] = None
    num_drifted_features: Optional[int] = None
    drift_report_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None