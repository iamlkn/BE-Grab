from pydantic import BaseModel, constr
from datetime import datetime
from typing import Optional
#from app.schemas.dataset import Dataset as DatasetSchema # Import the Dataset schema

# Shared properties
class ProjectBase(BaseModel):
    name: constr(min_length=1, max_length=255) # Enforce name constraints

# Properties to receive on project creation
class ProjectCreate(ProjectBase):
    dataset_id: int

# Properties to receive on project update
class ProjectUpdate(BaseModel):
    name: Optional[constr(min_length=1, max_length=255)] = None # Only name is updatable

# Properties to return to client
class Project(ProjectBase):
    id: int
    dataset_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True