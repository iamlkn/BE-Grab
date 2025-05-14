from uuid import UUID
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import datetime

from app.crud.base import CRUDBase
from app.db.models.automl_sessions import AutoMLSession
from app.schemas.automl_sessions import AutoMLSessionCreate, AutoMLSessionUpdateStepStatus # Use internal update schema

# Define Update Schema (can use the Pydantic one or Dict)
AutoMLSessionUpdateSchema = AutoMLSessionUpdateStepStatus

class CRUDAutoMLSession(CRUDBase[AutoMLSession, AutoMLSessionCreate, AutoMLSessionUpdateSchema]):

    def update_step_status(
        self,
        db: Session,
        *,
        session_id: int,
        step_number: int, # 1, 2, or 3
        status: str, # pending, running, completed, failed
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        **kwargs: Any # For other specific fields like paths, run_ids
    ) -> Optional[AutoMLSession]:
        """ Atomically updates the status and related fields for a specific step. """
        if step_number not in [1, 2, 3]:
            print(f"Error: Invalid step number {step_number} for status update.")
            return None # Or raise ValueError

        now = datetime.datetime.now(datetime.timezone.utc)
        values_to_update = {f"step{step_number}_status": status}

        if status == "running":
            values_to_update[f"step{step_number}_started_at"] = now
            values_to_update[f"step{step_number}_completed_at"] = None # Clear completion time
            values_to_update["error_message"] = None # Clear session error on start
        elif status in ["completed", "failed"]:
            values_to_update[f"step{step_number}_completed_at"] = now

        if results is not None:
             values_to_update[f"step{step_number}_results"] = results
        if error is not None:
             values_to_update["error_message"] = error # Update overall session error if step fails
        # Add specific result fields based on step and kwargs
        # Ensure kwargs keys match model columns (e.g., step1_experiment_path)
        valid_kwargs = {k: v for k, v in kwargs.items() if hasattr(self.model, k)}
        values_to_update.update(valid_kwargs)

        # Update overall status (simple logic, can be more complex)
        if status == "completed":
             if step_number == 1: values_to_update["overall_status"] = "step1_completed"
             elif step_number == 2: values_to_update["overall_status"] = "step2_completed"
             elif step_number == 3: values_to_update["overall_status"] = "step3_completed" # Or just "completed"
        elif status == "failed":
             values_to_update["overall_status"] = f"step{step_number}_failed" # Mark overall as failed

        # Use atomic update
        return self.update_atomic(db=db, id=session_id, values=values_to_update)

    def update_overall_status(
        self, db: Session, *, session_id: int, status: str, error: Optional[str] = None
        ) -> Optional[AutoMLSession]:
        """ Updates only the overall status and error message. """
        values = {"overall_status": status}
        if error is not None: values["error_message"] = error
        return self.update_atomic(db=db, id=session_id, values=values)


crud_automl_session = CRUDAutoMLSession(AutoMLSession)