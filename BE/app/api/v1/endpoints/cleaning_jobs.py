from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.schemas.cleaning_jobs import (
    CleaningConfig, CleaningPreview,
    CleaningStatus, CleaningResults, CleaningJobOut, CleaningDataPreview
)
# Assuming service functions handle underlying logic
from app.services.cleaning_service import (
    schedule_cleaning, run_cleaning_job,
    preview_cleaning, get_status, get_results,
    update_cleaning, delete_cleaning
)

from app.crud.cleaning_jobs import crud_cleaning_job # Used in post_cleaning
from app.crud.datasets import crud_dataset
from app.utils.file_storage import load_csv_as_dataframe, get_file_path
from app import schemas
import pandas as pd

# --- API Router Setup ---
router = APIRouter(prefix="/v1", tags=["cleaning"])

# --- Endpoint Definitions ---

@router.get("/datasets/{dataset_id}/cleaning/preview", response_model=CleaningPreview)
def preview_issues(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Preview Potential Data Quality Issues for a Dataset**

    Analyzes a dataset to identify potential data quality problems without modifying the data.

    This endpoint performs the following actions sequentially:
    1.  Delegates the analysis to the `preview_cleaning(dataset_id, db)` service function.
    2.  The service function is expected to:
        *   Load the dataset associated with `dataset_id`.
        *   Perform analysis (e.g., check missing values, data types, outliers).
        *   Format the findings into the `CleaningPreview` structure.
    3.  Returns the `CleaningPreview` object received from the service function.

    Args:
        `dataset_id` (int): The unique identifier of the dataset to preview.
        `db` (Session): **Dependency:** Injected database session.

    Returns:
        `CleaningPreview`: An object summarizing potential data quality issues.
    """
    # 1, 2, 3. Delegate to service layer
    return preview_cleaning(dataset_id, db)

@router.post("/datasets/{dataset_id}/cleaning", response_model=CleaningJobOut)
def post_cleaning(
    dataset_id: int,
    config: CleaningConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    **Schedule a New Data Cleaning Job**

    Creates a record for a data cleaning job based on the provided configuration and
    schedules the actual cleaning process to run asynchronously in the background.

    This endpoint performs the following actions sequentially:
    1.  Validates the request body (`config`) using Pydantic (`CleaningConfig`).
    2.  Calls the `schedule_cleaning(dataset_id, config.dict(), db)` service function. This function is expected to:
        *   Create a new cleaning job record in the database with a status like 'PENDING'.
        *   Return the unique ID (`job_id`) of the newly created job record.
    3.  Adds the `run_cleaning_job(job_id)` function to FastAPI's background tasks queue. This function will execute *after* the response is sent.
    4.  Retrieves the full details of the newly created job record from the database using `crud_cleaning_job.get(db, job_id)`.
    5.  Returns the details of the scheduled job (`CleaningJobOut`).

    Args:
        `dataset_id` (int): The ID of the dataset to be cleaned.
        `config` (CleaningConfig): Configuration parameters for the cleaning process.
        `background_tasks` (BackgroundTasks): **Dependency:** FastAPI utility for background tasks.
        `db` (Session): **Dependency:** Injected database session.

    Returns:
        `CleaningJobOut`: Details of the scheduled cleaning job, including its ID and initial status.
    """
    # 1. Validation (Implicit by Pydantic)
    # 2. Schedule the cleaning job via service layer
    job_id = schedule_cleaning(dataset_id, config.dict(), db)

    # 3. Add the actual cleaning task to run in the background
    background_tasks.add_task(run_cleaning_job, job_id)

    # 4. Retrieve the newly created job details
    job = crud_cleaning_job.get(db, job_id)
    crud_dataset.update_atomic(db, id=dataset_id, values={'is_clean':True})

    # 5. Return the job details
    return job

@router.get("/cleaning/{job_id}/status", response_model=CleaningStatus)
def cleaning_status(job_id: int, db: Session = Depends(get_db)):
    """
    **Get Cleaning Job Status**

    Retrieves the current processing status of a specific data cleaning job.

    This endpoint performs the following actions sequentially:
    1.  Calls the `get_status(job_id, db)` service function to retrieve the job's status string.
    2.  Validates that the returned status is not `None`. If it is `None`, raises a 404 error.
    3.  Constructs the `CleaningStatus` response object using the retrieved status string.
    4.  Returns the `CleaningStatus` object.

    Args:
        `job_id` (int): The unique identifier of the cleaning job.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`: `404` if a job with the specified `job_id` does not exist (service returned `None`).

    Returns:
        `CleaningStatus`: An object containing the current status string (e.g., "PENDING", "COMPLETED").
    """
    # 1. Get status via service layer
    status = get_status(job_id, db)

    # 2. Handle job not found
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # 3, 4. Return status
    return {"status": status}

@router.get("/cleaning/{job_id}/results", response_model=CleaningResults)
def cleaning_results(job_id: int, db: Session = Depends(get_db)):
    """
    **Get Cleaning Job Results Summary**

    Retrieves a summary of the results for a completed data cleaning job.

    This endpoint performs the following actions sequentially:
    1.  Calls the `get_results(job_id, db)` service function to retrieve the results summary dictionary.
    2.  Validates that the returned results are not `None`. If `None`, raises a 404 error (job not found or not completed).
    3.  Constructs the `CleaningResults` response object by combining the input `job_id` with the dictionary returned by the service.
    4.  Returns the `CleaningResults` object.

    Args:
        `job_id` (int): The unique identifier of the cleaning job.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`: `404` if results for the job are not found (job may not exist, not be complete, or failed).

    Returns:
        `CleaningResults`: An object containing the summary results (e.g., rows cleaned, values imputed).
    """
    # 1. Get results via service layer
    results = get_results(job_id, db)

    # 2. Handle results not found
    if results is None:
        raise HTTPException(status_code=404, detail="Results not found")

    # 3, 4. Return results
    return {"job_id": job_id, **results}


@router.get("/datasets/{dataset_id}/cleaned_data", response_model=CleaningDataPreview)
def get_cleaned_data(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Preview of Cleaned Data File**

    Retrieves a preview (typically the first 100 rows) of the *output file* generated
    by a cleaning job associated with the original `dataset_id`.

    This endpoint performs the following actions sequentially:
    1.  Constructs the expected file path for the *cleaned* data file (e.g., `storage/dataset_{id}_cleaned.csv`) using `get_file_path`.
    2.  Loads the entire cleaned dataset from the CSV file at the constructed path into a pandas DataFrame using `load_csv_as_dataframe`.
    3.  Determines the number of rows for the preview (minimum of 100 or total rows if fewer).
    4.  Extracts the first `preview_row` rows from the DataFrame using `.head()`.
    5.  Gets the total number of rows from the originally loaded DataFrame (`len(result_df)`).
    6.  Formats the preview DataFrame for JSON serialization:
        *   Resets the index.
        *   Converts the DataFrame to a list of lists using `.values.tolist()`.
        *   Replaces any `NaN`/`NaT` values with `None` using `.astype(object).where(pd.notnull(...), None)`.
    7.  Constructs a `schemas.DataFrameStructure` object containing the preview columns and data list.
    8.  Constructs the final `CleaningDataPreview` response object, including the preview structure, preview row count, and total row count.
    9.  Returns the `CleaningDataPreview` object.

    Args:
        `dataset_id` (int): The ID of the *original* dataset whose cleaned output is requested.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - Potentially (from `load_csv_as_dataframe`): If the expected cleaned data file (`dataset_{id}_cleaned.csv`) is not found or cannot be loaded.
            - `500`: If an error occurs during the formatting of the preview data for the JSON response.

    Returns:
        `CleaningDataPreview`: An object containing the cleaned data preview structure, preview row count, and total row count.
    """
    # 1. Construct file path for cleaned data
    file_path = get_file_path(f'dataset_{dataset_id}_cleaned.csv')

    # 2. Load cleaned data from file
    result_df = load_csv_as_dataframe(file_path)

    # 3. Determine preview rows
    preview_row = min(100, len(result_df))
    # 4. Extract preview DataFrame
    preview_df = result_df.head(preview_row).copy()
    # 5. Get total rows
    total_row = len(result_df)

    # 6, 7. Format preview for response
    try:
        df_for_schema = preview_df.reset_index(drop=True)
        data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
        preview_schema = schemas.DataFrameStructure(
            columns=df_for_schema.columns.tolist(),
            data=data_list
        )
    except Exception as convert_e:
        print(f"Error converting preview DataFrame to schema: {convert_e}")
        raise HTTPException(status_code=500, detail="Failed to format cleaned data preview.")

    # 8. Construct response object
    response_data = CleaningDataPreview(
        preview_cleaned=preview_schema,
        preview_row=preview_row,
        total_row=total_row
    )

    # 9. Return response
    return response_data


@router.put("/cleaning/{job_id}")
def put_cleaning(job_id: int, config: CleaningConfig, db: Session = Depends(get_db)):
    """
    **Update Cleaning Job Configuration**

    Updates the configuration of an existing cleaning job. The practical effect may
    depend on the job's current status (e.g., typically only applies if not yet started).

    This endpoint performs the following actions sequentially:
    1.  Validates the request body (`config`) using Pydantic (`CleaningConfig`).
    2.  Calls the `update_cleaning(job_id, config.dict(), db)` service function. This function attempts to update the job's configuration in the database.
    3.  Checks the boolean return value from the service function.
    4.  If the service function returns `False` (indicating the job was not found), raises a 404 error.
    5.  If the service function returns `True`, returns a success confirmation message.

    Args:
        `job_id` (int): The ID of the cleaning job to update.
        `config` (CleaningConfig): The *new* configuration settings.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`: `404` if a job with the specified `job_id` is not found.

    Returns:
        `dict`: Confirmation message `{"detail": "Config updated"}`.
    """
    # 1. Validation (Implicit)
    # 2. Attempt update via service layer
    success = update_cleaning(job_id, config.dict(), db)

    # 3, 4. Handle job not found
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    # 5. Return success message
    return {"detail": "Config updated"}

@router.delete("/cleaning/{job_id}")
def delete_clean(job_id: int, db: Session = Depends(get_db)):
    """
    **Delete Cleaning Job Record**

    Removes the metadata record of a specific cleaning job from the database.
    Note: This typically does not delete associated data files (e.g., the cleaned output).

    This endpoint performs the following actions sequentially:
    1.  Calls the `delete_cleaning(job_id, db)` service function. This function attempts to delete the job record from the database.
    2.  Checks the boolean return value from the service function.
    3.  If the service function returns `False` (indicating the job was not found), raises a 404 error.
    4.  If the service function returns `True`, returns a success confirmation message.

    Args:
        `job_id` (int): The ID of the cleaning job to delete.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`: `404` if a job with the specified `job_id` is not found.

    Returns:
        `dict`: Confirmation message `{"detail": "Job deleted"}`.
    """
    # 1. Attempt delete via service layer
    success = delete_cleaning(job_id, db)

    # 2, 3. Handle job not found
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    # 4. Return success message
    return {"detail": "Job deleted"}