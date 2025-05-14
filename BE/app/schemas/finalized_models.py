from pydantic import BaseModel, UUID4
from typing import Optional
import datetime

class FinalizedModelResponse(BaseModel): # Response schema
    id: int
    session_id: int # Matches AutoMLSession.id type
    model_name: str
    saved_model_path: str
    saved_metadata_path: str
    created_at: datetime.datetime
    # Add mlflow fields if they exist in the model
    mlflow_run_id: Optional[str] = None
    mlflow_registered_version: Optional[int] = None
    model_uri_for_registry: Optional[str] = None

    class Config:
        from_attributes = True
        
class FinalizedModelInfo(BaseModel):
    id: int
    session_id: int
    automl_session_name: str | None # Get name from related session
    model_name: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True 

# Add Create/Update schemas if you need to manage FinalizedModel directly via API
# Usually created internally by the finalize step service