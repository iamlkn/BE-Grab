from pydantic import BaseModel, UUID4
from typing import Optional, List, Dict, Any

class JobReference(BaseModel):
    job_id: int
    message: str

class StatusResponse(BaseModel):
    status: str
    error_message: Optional[str] = None

class DataFrameStructure(BaseModel):
    columns: List[str]
    data: List[List[Any]]