from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- Import Model and Base CRUD ---
from app.db.models.cleaning_jobs import CleaningJob
from app.crud.base import CRUDBase

# --- Schemas ---
# Define data structures for creating and updating CleaningJob records.

class CleaningJobCreateSchema(BaseModel):
    dataset_id: int
    config: Dict[str, Any] = {} # Configuration for the cleaning job
    status: str = "pending" # Default status on creation

class CleaningJobUpdateSchema(BaseModel):
    dataset_id: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    results: Optional[Dict[str, Any]] = None # Results populated after completion

# --- CRUD Class using CRUDBase ---
class CRUDCleaningJob(CRUDBase[CleaningJob, CleaningJobCreateSchema, CleaningJobUpdateSchema]):
    pass

# --- Instantiate the CRUD class ---
crud_cleaning_job = CRUDCleaningJob(CleaningJob)
