from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from app.schemas.commons import DataFrameStructure
from app.schemas.automl_jobs import AutoMLJobResultBase

class AnalyzeModelRequest(BaseModel):
    model_id_to_analyze: str = Field(...)
    model_name_for_logging: str = Field("analysis")
    return_raw_explanations: bool = Field(False)

class AnalysisResult(AutoMLJobResultBase):
    holdout_metrics: Optional[Dict[str, Any]] = None
    feature_importances: Optional[Union[DataFrameStructure, List[Dict[str, Any]], None]] = None
    shap_summary_plot_generated: Optional[bool] = None
    shap_artifact_path: Optional[str] = None