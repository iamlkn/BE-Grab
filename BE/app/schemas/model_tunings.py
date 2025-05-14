from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any
from app.schemas.commons import DataFrameStructure
from app.schemas.automl_jobs import AutoMLJobResultBase

class TuneModelRequest(BaseModel):
    model_id: Optional[str] = None
    use_best_from_compare: Optional[bool] = False
    optimize_metric: Optional[str] = None
    n_iter: Optional[int] = Field(10, gt=0)
    custom_grid: Optional[Dict[str, Any]] = None

    @model_validator(pre=True)
    def check_model_source(cls, values):
        model_id, use_best = values.get('model_id'), values.get('use_best_from_compare')
        if not (model_id or use_best): raise ValueError('Either model_id or use_best_from_compare=True must be provided')
        if model_id and use_best: raise ValueError('Provide either model_id or use_best_from_compare=True, not both')
        return values

class TuneModelResult(AutoMLJobResultBase):
    tuned_model_id: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None
    cv_metrics: Optional[Dict[str, Any]] = None
    cv_metrics_dataframe: Optional[DataFrameStructure] = None