import logging
from sqlalchemy.orm import Session
from app.crud.datasets import crud_dataset
from app.crud.cleaning_jobs import crud_cleaning_job
from app.crud.datasets import DatasetCreateSchema # Assuming schema is in crud file
from app.crud.cleaning_jobs import CleaningJobCreateSchema, CleaningJobUpdateSchema # Assuming schemas are in crud file
from app.utils.cleaning_tool import automate_cleaning_by_task
from app.utils.file_storage import load_csv_as_dataframe
from app.db.session import SessionLocal
from app.utils.file_storage import get_file_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def schedule_cleaning(dataset_id: int, config: dict, db: Session) -> int:
    """Schedules a new cleaning job using CRUDBase."""
    logger.info(f"Scheduling cleaning job for dataset_id: {dataset_id}")
    job_in = CleaningJobCreateSchema(dataset_id=dataset_id, config=config, status='pending')
    job = crud_cleaning_job.create(db=db, obj_in=job_in)
    logger.info(f"Created cleaning job with id: {job.id}")
    return job.id


def run_cleaning_job(job_id: int):
    """Executes the cleaning process for a given job ID using CRUDBase."""
    db: Session | None = None
    logger.info(f"Attempting to run cleaning job_id: {job_id}")
    try:
        db = SessionLocal() # Create a new session for this background task

        # Fetch job using CRUDBase
        job = crud_cleaning_job.get(db=db, id=job_id)
        if not job:
            logger.error(f"Cleaning job_id: {job_id} not found.")
            return # Exit if job doesn't exist

        # Prevent re-running completed/failed jobs (optional check)
        if job.status not in ['pending', 'failed']: # Allow re-running failed jobs maybe? Adjust logic as needed.
             logger.warning(f"Cleaning job_id: {job_id} is not in 'pending'/'failed' state (current: {job.status}). Skipping run.")
             return

        # Update status to 'running' using CRUDBase
        job_update_running = CleaningJobUpdateSchema(status='running')
        job = crud_cleaning_job.update(db=db, db_obj=job, obj_in=job_update_running)
        logger.info(f"Set job_id: {job_id} status to 'running'.")

        # Fetch dataset using CRUDBase
        ds = crud_dataset.get(db=db, id=job.dataset_id)
        if not ds:
            logger.error(f"Dataset id: {job.dataset_id} associated with job_id: {job_id} not found.")
            # Update job status to failed
            job_update_fail = CleaningJobUpdateSchema(status='failed', results={"error": "Original dataset not found"})
            crud_cleaning_job.update(db=db, db_obj=job, obj_in=job_update_fail)
            return

        original = ds.file_path
        # Define output path based on original dataset and job ID
        filename = f"dataset_{ds.id}_cleaned.csv"
        output_path = get_file_path(filename)

        logger.info(f"Starting cleaning process for job_id: {job_id}. Input: {original}, Output: {output_path}")
        # Run the actual cleaning task
        automate_cleaning_by_task(
            input_csv=original,
            output_csv=output_path,
            remove_duplicates=job.config.get('remove_duplicates', False),
            handle_missing_values=job.config.get('handle_missing_values', False),
            smooth_noisy_data=job.config.get('smooth_noisy_data', False),
            handle_outliers=job.config.get('handle_outliers', False),
            reduce_cardinality=job.config.get('reduce_cardinality', False),
            encode_categorical_values=job.config.get('encode_categorical_values', False),
            feature_scaling=job.config.get('feature_scaling', False),
        )
        logger.info(f"Cleaning process completed for job_id: {job_id}.")


        # Calculate results
        try:
            df_orig = load_csv_as_dataframe(original)
            df_clean = load_csv_as_dataframe(output_path)
            results = {
                'original_rows': len(df_orig),
                'cleaned_rows': len(df_clean),
                'cleaned_dataset_id': ds.id,
                'cleaned_file_path': output_path, # Store the path for reference
                'error': None # Explicitly set error to None on success
            }
        except Exception as e:
             logger.error(f"Error loading dataframes for results calculation (job_id: {job_id}): {e}")
             results = {
                 'original_rows': -1, # Indicate error
                 'cleaned_rows': -1,
                 'cleaned_dataset_id': ds.id, # Still store the ID even if stats failed
                 'cleaned_file_path': output_path,
                 'error': f"Failed to calculate statistics: {e}"
             }

        # Update job status to 'completed' with results using CRUDBase
        job_update_done = CleaningJobUpdateSchema(status='completed', results=results)
        job = crud_cleaning_job.update(db=db, db_obj=job, obj_in=job_update_done)
        logger.info(f"Set job_id: {job_id} status to 'completed'.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred during run_cleaning_job for job_id: {job_id}: {e}")
        if db and 'job' in locals() and job: # Check if db and job exist before trying to update
            try:
                 # Attempt to mark the job as failed if an error occurred
                job_update_fail = CleaningJobUpdateSchema(status='failed', results={"error": str(e)})
                crud_cleaning_job.update(db=db, db_obj=job, obj_in=job_update_fail)
                logger.info(f"Set job_id: {job_id} status to 'failed' due to error.")
            except Exception as update_err:
                 logger.error(f"Failed to update job {job_id} status to 'failed': {update_err}")
       
    finally:
        if db:
            db.close()
            logger.debug(f"Database session closed for job_id: {job_id}.")


def preview_cleaning(dataset_id: int, db: Session) -> dict:
    """Generates a preview of potential cleaning issues using CRUDBase."""
    logger.debug(f"Generating cleaning preview for dataset_id: {dataset_id}")
    # Fetch dataset using CRUDBase
    ds = crud_dataset.get(db=db, id=dataset_id)
    if not ds:
        logger.warning(f"Dataset_id: {dataset_id} not found for preview.")
        # Return empty or raise error depending on desired API behavior
        return {'error': 'Dataset not found'}

    try:
        df = load_csv_as_dataframe(ds.file_path)
        missing = int(df.isna().sum().sum())
        outliers_per_col = {
            col: int(((df[col] - df[col].mean()).abs() > 3 * df[col].std()).sum())
            for col in df.select_dtypes(include="number")
        }
        outliers = sum(outliers_per_col.values())
        duplicates = int(df.duplicated().sum())
        logger.debug(f"Preview generated successfully for dataset_id: {dataset_id}")
        return {'missing': missing, 'outliers': outliers, 'duplicates': duplicates}
    except Exception as e:
        logger.error(f"Error generating preview for dataset_id {dataset_id}: {e}")
        return {'error': f'Failed to generate preview: {e}'}


def get_status(job_id: int, db: Session) -> str | None:
    """Gets the status of a cleaning job using CRUDBase."""
    logger.debug(f"Fetching status for job_id: {job_id}")
    job = crud_cleaning_job.get(db=db, id=job_id)
    if not job:
        logger.warning(f"Job_id: {job_id} not found when fetching status.")
        return None # Or raise HTTPException(404) in an API context
    return job.status


def get_results(job_id: int, db: Session) -> dict | None:
    """Gets the results of a completed cleaning job using CRUDBase."""
    logger.debug(f"Fetching results for job_id: {job_id}")
    job = crud_cleaning_job.get(db=db, id=job_id)
    if not job:
        logger.warning(f"Job_id: {job_id} not found when fetching results.")
        return None # Or raise HTTPException(404)
    if job.status != 'completed':
        return {"status": job.status, "message": "Job not completed yet"}
    return job.results


def update_cleaning(job_id: int, config: dict, db: Session) -> bool:
    """Updates the configuration of a pending cleaning job using CRUDBase."""
    logger.info(f"Attempting to update config for job_id: {job_id}")
    job = crud_cleaning_job.get(db=db, id=job_id)
    if not job:
        logger.warning(f"Job_id: {job_id} not found for update.")
        return False

    if job.status != 'pending':
        logger.warning(f"Cannot update config for job_id: {job_id} because its status is '{job.status}'.")
        return False

    job_update_config = CleaningJobUpdateSchema(config=config)
    updated_job = crud_cleaning_job.update(db=db, db_obj=job, obj_in=job_update_config)
    logger.info(f"Config updated successfully for job_id: {job_id}")
    return updated_job is not None


def delete_cleaning(job_id: int, db: Session) -> bool:
    """Deletes a cleaning job using CRUDBase."""
    logger.info(f"Attempting to delete job_id: {job_id}")
    # CRUDBase.remove returns the deleted object or None
    deleted_job = crud_cleaning_job.remove(db=db, id=job_id)
    if deleted_job:
        logger.info(f"Successfully deleted job_id: {job_id}")
        return True
    else:
        logger.warning(f"Job_id: {job_id} not found for deletion.")
        return False