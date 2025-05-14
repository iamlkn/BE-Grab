from pydantic import BaseModel, Field, Extra
from typing import List, Dict, Any, Union, Optional

# --- Schemas for input data structures ---

class SummaryStatItem(BaseModel):
    column: str
    count: int
    unique: Union[str, int, None] = None # Allow empty string or actual int
    top: Union[str, int, float, None] = None
    freq: Union[str, int, float, None] = None
    mean: Union[str, float, None] = None
    std: Union[str, float, None] = None
    min: Union[str, int, float, None] = None
    percent_25: Union[str, float, None] = Field(None, alias="25%")
    percent_50: Union[str, float, None] = Field(None, alias="50%")
    percent_75: Union[str, float, None] = Field(None, alias="75%")
    max: Union[str, int, float, None] = None

    class Config:
        populate_by_name = True # Allows using "25%" in JSON

class SummaryStatsInput(BaseModel):
    data: List[SummaryStatItem]

class CorrelationMatrixInput(BaseModel):
    data: Dict[str, Dict[str, float]]

class ModelPerformanceInput(BaseModel):
    columns: List[str]
    data: List[List[Union[str, float, int]]]

class TunedModelBestParams(BaseModel):
    copy_X: Optional[bool] = None
    fit_intercept: Optional[bool] = None
    n_jobs: Optional[int] = None
    positive: Optional[bool] = None

class TunedModelCVMetricsRow(BaseModel):
    fold_or_stat: Union[str, int] = Field(..., alias="Fold") # To match "Fold" or "Mean"/"Std"
    # Regression metrics (optional)
    mae: Optional[float] = Field(None, alias="MAE")
    mse: Optional[float] = Field(None, alias="MSE")
    rmse: Optional[float] = Field(None, alias="RMSE")
    r2: Optional[float] = Field(None, alias="R2")
    rmsle: Optional[float] = Field(None, alias="RMSLE")
    mape: Optional[float] = Field(None, alias="MAPE")

    # Classification metrics (optional)
    accuracy: Optional[float] = Field(None, alias="Accuracy")
    auc: Optional[float] = Field(None, alias="AUC")
    f1: Optional[float] = Field(None, alias="F1")
    precision: Optional[float] = Field(None, alias="Precision")
    recall: Optional[float] = Field(None, alias="Recall")

    # Common metric
    tt: Optional[float] = Field(None, alias="TT (Sec)")

    model_config = {
        "populate_by_name": True,
        "extra": "allow"  # This allows you to include any other metric like "LogLoss", "NDCG", etc.
    }
        
class BaselineModelMetricsData(BaseModel):
    # Regression metrics (optional)
    mae: Optional[float] = Field(None, alias="MAE")
    mse: Optional[float] = Field(None, alias="MSE")
    rmse: Optional[float] = Field(None, alias="RMSE")
    r2: Optional[float] = Field(None, alias="R2")
    rmsle: Optional[float] = Field(None, alias="RMSLE")
    mape: Optional[float] = Field(None, alias="MAPE")

    # Classification metrics (optional)
    accuracy: Optional[float] = Field(None, alias="Accuracy")
    auc: Optional[float] = Field(None, alias="AUC")
    f1: Optional[float] = Field(None, alias="F1")
    precision: Optional[float] = Field(None, alias="Precision")
    recall: Optional[float] = Field(None, alias="Recall")

    # Common metric
    tt: Optional[float] = Field(None, alias="TT (Sec)")

    model_config = {
        "populate_by_name": True,
        "extra": "allow"  # This allows you to include any other metric like "LogLoss", "NDCG", etc.
    }

class TunedModelCVMetricsTable(BaseModel):
    columns: List[str]
    data: List[List[Union[str, float, int]]]

class TunedModelResultsData(BaseModel):
    best_params: Dict[str, Any] # Using Dict[str, Any] for flexibility as in example
    cv_metrics_table: Dict[str, Any]

class TunedModelInput(BaseModel):
    tuning_data: TunedModelResultsData
    image_url: Optional[str] = None # For providing a public URL to an image

# --- Response Schema ---
class AISummaryResponse(BaseModel):
    summary_html: str
    input_type: str