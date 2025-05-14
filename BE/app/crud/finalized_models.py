from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.crud.base import CRUDBase
from app.db.models.finalized_models import FinalizedModel

# Internal schema for creation - matches new model structure
class FinalizedModelCreateInternal(BaseModel):
    session_id: int # Changed to int
    # automl_job_id: Optional[int] = None # Removed job link
    model_name: str
    saved_model_path: str
    saved_metadata_path: str
    # Add mlflow fields if needed by create logic
    mlflow_run_id: Optional[str] = None
    model_uri_for_registry: Optional[str] = None


# Update schema remains Dict or specific Pydantic model if needed
FinalizedModelUpdateSchema = Dict[str, Any]

class CRUDFinalizedModel(CRUDBase[FinalizedModel, FinalizedModelCreateInternal, FinalizedModelUpdateSchema]):
     def update_registration(
         self, db: Session, *, model_id: int, mlflow_version: int, model_uri: Optional[str] = None
     ) -> FinalizedModel | None:
         # Logic remains similar, update atomic handles column check
         values = {"mlflow_registered_version": mlflow_version}
         if model_uri: values["model_uri_for_registry"] = model_uri
         return self.update_atomic(db=db, id=model_id, values=values)

     # Add get by session_id if needed
     def get_by_session_id(self, db: Session, *, session_id: int) -> Optional[FinalizedModel]:
          return db.query(self.model).filter(self.model.session_id == session_id).first()


crud_finalized_model = CRUDFinalizedModel(FinalizedModel)