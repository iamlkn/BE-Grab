from pydantic import BaseModel, Field, model_validator, UUID4
from typing import Optional, Dict, Any

class RegisterModelRequest(BaseModel):
    finalized_model_id: Optional[int] = None
    model_uri: Optional[str] = None
    registered_model_name: str = Field(...)
    await_registration_seconds: int = Field(10, ge=0)
    description: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None

    @model_validator(pre=True)
    def check_model_source(cls, values):
        fm_id, uri = values.get('finalized_model_id'), values.get('model_uri')
        if not (fm_id or uri) or (fm_id and uri): 
            raise ValueError("Provide exactly one of 'finalized_model_id' or 'model_uri'")
        return values

class RegisterModelResponse(BaseModel):
    name: str
    version: str
    creation_timestamp: int
    last_updated_timestamp: Optional[int] = None
    current_stage: str
    description: Optional[str] = None
    source: Optional[str] = None
    run_id: Optional[str] = None
    status: Optional[str] = None
    status_message: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    run_link: Optional[str] = None