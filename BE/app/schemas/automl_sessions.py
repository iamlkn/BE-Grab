from pydantic import BaseModel, Field, UUID4
from typing import Optional, List, Dict, Any
import datetime
from app.schemas.commons import DataFrameStructure, StatusResponse # Import common schemas

# --- Base and Create ---
class AutoMLSessionBase(BaseModel):
    name: Optional[str] = None
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None

class AutoMLSessionCreate(AutoMLSessionBase):
    # No extra fields needed just to create the initial record
    pass

# --- Step 1 Specific Schemas ---
class AutoMLSessionStartStep1Request(BaseModel): # Rename for clarity
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None
    name: Optional[str] = None
    # Include config overrides if needed, e.g.
    # config_overrides: Optional[Dict[str, Any]] = None
    
class AutoMLSessionStep1Response(BaseModel): # The missing class
    session_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None
    status: str # Should be "step1_completed" on success
    # You can embed the results directly or use the nested class:
    task_type: Optional[str]
    experiment_save_path: Optional[str]
    comparison_results: Optional[DataFrameStructure] = None
    # Or: step1_results: Optional[AutoMLSessionStep1Result] = None # Alternative using nested
    error_message: Optional[str] = None # Should be None on success

class AutoMLSessionStep1Result(BaseModel): # Specific results for Step 1
    task_type: Optional[str]
    experiment_save_path: Optional[str]
    comparison_results: Optional[DataFrameStructure] = None
    # Add other relevant results from step 1, e.g., best model ID?
    best_model_id: Optional[str] = None

# --- Step 2 Specific Schemas ---
class AutoMLSessionStartStep2Request(BaseModel):
    model_id_to_tune: str
    # Add tuning config overrides if needed
    # tune_config_overrides: Optional[Dict[str, Any]] = None

class AutoMLSessionStep2Result(BaseModel):
    tuned_model_id: str # The ID of the model that was tuned (e.g., 'rf')
    tuned_model_save_path_base: str # Base path where the tuned model was saved
    best_params: Optional[Dict[str, Any]] = None # Dictionary of best hyperparameters found
    cv_metrics_table: Optional[DataFrameStructure] = None # CV metrics as a table structure
    feature_importance_plot_path: Optional[str] = None # Path to the saved feature importance plot

# --- Step 3 Specific Schemas ---
class AutoMLSessionStartStep3Request(BaseModel):
    # Input is typically the result of Step 2 (tuned model path)
    # Add finalize config overrides if needed
    # finalize_config_overrides: Optional[Dict[str, Any]] = None
    model_name_override: Optional[str] = None # Optional name for final model

class AutoMLSessionStep3Result(BaseModel):
    finalized_model_db_id: Optional[int] # ID from FinalizedModel table
    saved_model_path: Optional[str]
    saved_metadata_path: Optional[str]
    # Add other finalization results if any

# --- General Session Response ---
class AutoMLSessionResponse(BaseModel):
    # Core Info
    id: int
    name: Optional[str]
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]]
    config: Optional[Dict[str, Any]] # Show the config used

    # Overall Status
    overall_status: str
    error_message: Optional[str]

    # Step 1 Details
    step1_status: Optional[str]
    step1_results: Optional[AutoMLSessionStep1Result] # Embed step 1 results
    step1_mlflow_run_id: Optional[str]
    task_type: Optional[str] # Populate from step 1

    # Step 2 Details
    step2_status: Optional[str]
    step2_model_id_tuned: Optional[str]
    step2_results: Optional[AutoMLSessionStep2Result] # Embed step 2 results
    step2_mlflow_run_id: Optional[str]

    # Step 3 Details
    step3_status: Optional[str]
    step3_results: Optional[AutoMLSessionStep3Result] # Embed step 3 results
    step3_mlflow_run_id: Optional[str]

    # Timestamps
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime]

    class Config:
        from_attributes = True # Use orm_mode for Pydantic v1

# Error response remains the same
class AutoMLSessionErrorResponse(BaseModel):
    detail: str

# --- Schemas for internal updates (optional but can be useful) ---
class AutoMLSessionUpdateStepStatus(BaseModel):
    # Used by service to update status fields
    step1_status: Optional[str] = None
    step1_started_at: Optional[datetime.datetime] = None
    step1_completed_at: Optional[datetime.datetime] = None
    step1_results: Optional[Dict[str, Any]] = None # Store raw dict
    step1_experiment_path: Optional[str] = None
    step1_mlflow_run_id: Optional[str] = None
    task_type: Optional[str] = None # Update task type

    step2_status: Optional[str] = None
    step2_started_at: Optional[datetime.datetime] = None
    step2_completed_at: Optional[datetime.datetime] = None
    step2_model_id_tuned: Optional[str] = None # Track which model was tuned
    step2_results: Optional[Dict[str, Any]] = None
    step2_tuned_model_path_base: Optional[str] = None
    step2_mlflow_run_id: Optional[str] = None

    step3_status: Optional[str] = None
    step3_started_at: Optional[datetime.datetime] = None
    step3_completed_at: Optional[datetime.datetime] = None
    step3_final_model_id: Optional[UUID4] = None # Store FK to finalized model
    step3_mlflow_run_id: Optional[str] = None

    overall_status: Optional[str] = None
    error_message: Optional[str] = None # Clear or set error message
    config: Optional[Dict[str, Any]] = None # Allow updating stored config
    
class AutoMLSessionResultsDetail(BaseModel):
    session_id: int = Field(..., alias="id")
    step1_results: Optional[Dict[str, Any]] = None
    step2_results: Optional[Dict[str, Any]] = None
    step3_results: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True