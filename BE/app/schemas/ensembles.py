from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.schemas.commons import DataFrameStructure
from app.schemas.automl_jobs import AutoMLJobResultBase

class CreateEnsembleRequest(BaseModel):
    base_model_id: str = Field(...)
    method: str = Field("Bagging")
    n_estimators: Optional[int] = Field(10, gt=0)

class EnsembleResult(AutoMLJobResultBase):
    ensemble_model_id: Optional[str] = None
    cv_metrics: Optional[Dict[str, Any]] = None
    cv_metrics_dataframe: Optional[DataFrameStructure] = None