from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.schemas.commons import DataFrameStructure
from app.schemas.automl_jobs import AutoMLJobResultBase

class CompareModelsRequest(BaseModel):
    sort_metric: Optional[str] = None
    include_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    baseline_model_ids_for_holdout: List[str] = Field([])

class CompareModelsResult(AutoMLJobResultBase):
    best_model_id: Optional[str] = None
    comparison_results: Optional[DataFrameStructure] = None
    baseline_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    artifacts: Optional[Dict[str, str]] = None