from fastapi import APIRouter, Depends, HTTPException, status, Body, Path, UploadFile, File
from sqlalchemy.orm import Session # Sync Session
from uuid import UUID
from typing import Any # For Any response model
from app import schemas
from app.services import automl_service
from app.api.v1.dependencies import get_db
from starlette.responses import StreamingResponse # Needed for CSV response
import io # Needed for creating in-memory file for response
from app.crud.finalized_models import crud_finalized_model
from app.schemas.finalized_models import FinalizedModelInfo
from app.crud.datasets import crud_dataset
from app.db.models import AutoMLSession, FinalizedModel, Dataset
from typing import List
import pandas as pd
import base64
from sqlalchemy.ext.asyncio import AsyncSession # Assuming async setup
from sqlalchemy.future import select # Use future select for modern SQLAlchemy
from sqlalchemy.orm import joinedload, selectinload # Efficiently load related data
from starlette.responses import FileResponse # To send the file
import os


# Define the API router
router = APIRouter(
    prefix='/v1',
    tags=['modeling']
)

# --- Endpoint to Start Step 1 ---
@router.post(
    '/automl_sessions', # POST to the base path of this router (e.g., /api/v1/automl-sessions)
    response_model=schemas.AutoMLSessionStep1Response, # Use the specific Step 1 SUCCESS response schema
    status_code=status.HTTP_200_OK, # 200 OK because it's synchronous and returns results
    summary="Start AutoML Step 1: Setup & Compare (Sync & Blocking)",
    responses={ # Define potential error responses using the specific error schema
        status.HTTP_404_NOT_FOUND: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "Dataset specified by dataset_id not found in the database."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "AutoML Step 1 failed during execution or an internal server error occurred."
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "Invalid input parameters (e.g., missing target column)."
        },
    }
)
def start_automl_step1_endpoint(
    *,
    db: Session = Depends(get_db), # Inject synchronous DB Session
    params: schemas.AutoMLSessionStartStep1Request = Body(...) # Expect request body matching this schema
):
    """
    Initiates the synchronous AutoML Step 1 (Setup & Compare Models).

    This endpoint performs the following actions sequentially:
    1. Validates the input request body.
    2. Creates an initial record for this AutoML session in the database.
    3. Retrieves the file path associated with the provided `dataset_id`.
    4. Loads the base AutoML configuration (`config.yaml`).
    5. Merges the base configuration with request parameters and application settings.
    6. Instantiates the `AutoMLRunner`.
    7. **BLOCKS** while executing the `step1_setup_and_compare` method of the runner.
    8. Updates the AutoML session record in the database with the final status and results.
    9. Returns the results of Step 1, including the detected task type and comparison results.

    **Request Body:**
    - `dataset_id` (integer, required): ID of the dataset registered in the system.
    - `target_column` (string, required): Name of the column to predict.
    - `feature_columns` (array[string], optional): List of features to use. Uses all others if omitted.
    - `name` (string, optional): Optional descriptive name for this AutoML session.
    """
    try:
        # Call the service function which contains all the logic
        result = automl_service.run_step1_setup_and_compare(db, params)
        # The service function returns the validated AutoMLSessionStep1Response object on success
        crud_dataset.update_atomic(db, id=params.dataset_id, values={'is_model':True})
        return result
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like 404 Not Found, 400 Bad Request, 500 Internal Server Error)
        # that might be raised explicitly within the service layer or its dependencies.
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors that weren't converted to HTTPException
        # Log the error for debugging
        print(f"FATAL: Unexpected error in Step 1 endpoint: {e}")
        # Return a generic 500 error to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during AutoML Step 1: {e}"
        )

@router.post(
    "/{session_id}/step2-tune", # Define a path for step 2, includes session_id
    response_model=schemas.AutoMLSessionStep2Result, # Use the specific Step 2 SUCCESS response
    status_code=status.HTTP_200_OK, # Sync success
    summary="Start AutoML Step 2: Tune & Analyze Model (Sync & Blocking)",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": schemas.AutoMLSessionErrorResponse, "description": "Session or required Step 1 artifacts not found"},
        status.HTTP_400_BAD_REQUEST: {"model": schemas.AutoMLSessionErrorResponse, "description": "Invalid input or prerequisite Step 1 not completed"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": schemas.AutoMLSessionErrorResponse, "description": "AutoML Step 2 failed or Internal Error"}
    }
)
def start_automl_step2_endpoint(
    *,
    session_id: int, # Path parameter for the session
    db: Session = Depends(get_db),
    params: schemas.AutoMLSessionStartStep2Request = Body(...) # Body contains model_id to tune
):
    """
    Initiates the synchronous AutoML Step 2 (Tune & Analyze Model) for a given session.

    - Loads the experiment saved in Step 1.
    - Tunes the specified `model_id_to_tune` based on configuration.
    - Analyzes the tuned model (e.g., plots, SHAP).
    - Saves the tuned model artifact.
    - Updates the Step 2 status and results in the AutoML Session record.
    - **BLOCKS** until Step 2 completes or fails.

    **Path Parameter:**
    - `session_id` (integer): The ID of the AutoML Session created in Step 1.

    **Request Body:**
    - `model_id_to_tune` (string, required): The PyCaret model ID (e.g., 'rf', 'lightgbm') to be tuned. Must be one of the models compared in Step 1.
    """
    try:
        # Call the NEW service function for Step 2
        result = automl_service.run_step2_tune_and_analyze(db, session_id, params)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"FATAL: Unexpected error in Step 2 endpoint for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during AutoML Step 2: {e}"
        )
        
        
@router.post(
    "/{session_id}/step3-finalize", # Define path for step 3
    response_model=schemas.AutoMLSessionStep3Result, # Use Step 3 success response
    status_code=status.HTTP_200_OK, # Sync success
    summary="Start AutoML Step 3: Finalize Model (Sync & Blocking)",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": schemas.AutoMLSessionErrorResponse, "description": "Session or required Step 1/2 artifacts not found"},
        status.HTTP_400_BAD_REQUEST: {"model": schemas.AutoMLSessionErrorResponse, "description": "Invalid input or prerequisite Step 1/2 not completed"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": schemas.AutoMLSessionErrorResponse, "description": "AutoML Step 3 failed or Internal Error"}
    }
)
def start_automl_step3_endpoint(
    *,
    session_id: int, # Path parameter for the session
    db: Session = Depends(get_db),
    params: schemas.AutoMLSessionStartStep3Request = Body(None, description="Optional parameters like model name override.") # Body is optional for this step currently
):
    """
    Initiates the synchronous AutoML Step 3 (Finalize Model).

    - Loads the experiment from Step 1 and the tuned model from Step 2.
    - Finalizes the tuned model (trains on full dataset).
    - Saves the final model artifact.
    - Creates a `FinalizedModel` record in the database.
    - Updates the Step 3 status and results in the AutoML Session record, linking to the `FinalizedModel`.
    - **BLOCKS** until Step 3 completes or fails.

    **Path Parameter:**
    - `session_id` (integer): The ID of the AutoML Session where Steps 1 & 2 completed.

    **Request Body (Optional):**
    - `model_name_override` (string, optional): Provide a specific name for the saved final model artifact.
    """
    try:
        request_params = params if params is not None else schemas.AutoMLSessionStartStep3Request()

        # Call function for Step 3
        result = automl_service.run_step3_finalize_and_save(db, session_id, request_params)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"FATAL: Unexpected error in Step 3 endpoint for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during AutoML Step 3: {e}"
        )

@router.post(
    "/finalized_models/{finalized_model_id}/predict",
    # --- Use the new JSON response schema ---
    response_model=schemas.PredictionResponse,
    summary="Predict from Uploaded CSV and Return Preview (Sync & Blocking)",
    status_code=status.HTTP_200_OK,
    responses={
        # --- Describe the new JSON response ---
        status.HTTP_200_OK: {
            "model": schemas.PredictionResponse,
            "description": "Returns a JSON object containing a preview (first 10 rows) of the predictions.",
        },
        status.HTTP_404_NOT_FOUND: {"model": schemas.AutoMLSessionErrorResponse, "description": "Finalized Model or associated data not found"},
        status.HTTP_400_BAD_REQUEST: {"model": schemas.AutoMLSessionErrorResponse, "description": "Invalid, empty, or unparseable CSV file"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": schemas.AutoMLSessionErrorResponse, "description": "Prediction failed or Internal Error"}
    }
)
async def predict_with_finalized_model_csv_endpoint( # Still async for file reading
    *,
    finalized_model_id: int = Path(..., description="The ID of the finalized model to use."),
    db: Session = Depends(get_db),
    file: UploadFile = File(..., description="CSV file containing the data to predict on.")
):
    """
    Makes predictions by uploading a CSV file to a finalized AutoML model
    and returns a JSON response with a **preview** of the first 10 rows.

    - Reads the uploaded CSV file.
    - Fetches the specified finalized model artifact path and configuration.
    - Loads the model pipeline.
    - Applies the pipeline to the data from the CSV.
    - Optionally performs data drift check (if configured in the runner).
    - Returns a **JSON object** containing the first 10 rows (or fewer) of the
      original data plus the prediction columns.
    - **BLOCKS** until prediction completes or fails.
    - **Does not store** the uploaded data or prediction results in the database.
    - **Does not return** the full prediction results file via this endpoint.

    **Path Parameter:**
    - `finalized_model_id` (integer): The unique ID of the `FinalizedModel` record.

    **File Upload:**
    - `file`: The request must contain a file upload with the key `file`, holding the CSV data.
             The CSV header must match the feature columns used during model training.
    """
    # Basic check for CSV MIME type (optional but recommended)
    if file.content_type not in ['text/csv', 'application/vnd.ms-excel', 'text/plain']: # Allow text/plain sometimes
         file.close() # Close the file before raising
         raise HTTPException(
             status_code=status.HTTP_400_BAD_REQUEST,
             detail=f"Invalid file type. Please upload a CSV file. Received: {file.content_type}"
         )

    result_df = None # Initialize
    full_csv_base64_str = None
    try:
        # Call the service function - it returns the FULL DataFrame
        result_df = await automl_service.run_prediction_from_csv(db, finalized_model_id, file) # File is closed inside service

        # --- Generate Preview ---
        preview_rows = min(5, len(result_df))
        preview_df = result_df.head(preview_rows).copy() # Take first N rows
        total_rows = len(result_df)

        # --- Convert Preview DataFrame to Schema Format ---
        try:
            # Ensure index is reset if needed, handle potential NaNs for JSON conversion
            df_for_schema = preview_df.reset_index(drop=True)
            data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
            preview_schema = schemas.DataFrameStructure(
                columns=df_for_schema.columns.tolist(),
                data=data_list
            )
        except Exception as convert_e:
            # Log the error internally if needed
            print(f"Error converting preview DataFrame to schema: {convert_e}")
            raise HTTPException(status_code=500, detail="Failed to format prediction preview results.")
        
        # --- Generate Full CSV String and Base64 Encode ---
        try:
            stream = io.StringIO()
            result_df.to_csv(stream, index=False)
            stream.seek(0)
            csv_string = stream.getvalue()
            # Encode: string -> bytes (utf-8) -> base64 bytes -> string (utf-8)
            base64_bytes = base64.b64encode(csv_string.encode('utf-8'))
            full_csv_base64_str = base64_bytes.decode('utf-8')
            stream.close() # Close the string buffer
        except Exception as encode_e:
            print(f"Error encoding full CSV to base64: {encode_e}")
            # Decide if this is fatal or just skip the full csv
            full_csv_base64_str = None # Or raise HTTPException

        # --- Fetch session_id (needed for response schema) ---
        # We could refactor the service to return it, or fetch it again briefly
        db_finalized_model = crud_finalized_model.get(db, id=finalized_model_id) # Assumes model still exists
        session_id = db_finalized_model.session_id if db_finalized_model else -1 # Fallback

        # --- Construct the JSON Response ---
        response_data = schemas.PredictionResponse(
            session_id=session_id,
            finalized_model_id=finalized_model_id,
            preview_predictions=preview_schema,
            total_rows_processed=total_rows,
            full_csv_base64=full_csv_base64_str
            # message is set by default in the schema
        )
        return response_data

    except HTTPException as http_exc:
        # Re-raise known HTTP errors (e.g., file read errors, model not found)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors during processing or preview generation
        print(f"FATAL: Unexpected error in CSV prediction endpoint for model {finalized_model_id}: {e}")
        # Ensure file is closed if error happened before service call finished
        if file and hasattr(file, 'file') and not file.file.closed:
            print(f"Closing file: {file.filename}")
            file.close()
        elif file and hasattr(file, 'file') and file.file.closed:
            print(f"File already closed: {file.filename}")
        elif file:
            print(f"File object {file.filename} exists but has no '.file' attribute?")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during CSV prediction processing: {e}"
        )
        
@router.get(
    "/datasets/{dataset_id}/automl-session-results/",
    response_model=schemas.automl_sessions.AutoMLSessionResultsDetail, 
    status_code=status.HTTP_200_OK,
    summary="Get AutoML Session Step Results for a Dataset",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "Dataset not found."
        }
    }
)
def get_automl_session_results_for_dataset(
    dataset_id: int = Path(..., title="The ID of the dataset to retrieve session results for"),
    db: Session = Depends(get_db)
):
    """
    Retrieves step1, step2, and step3 results along with other key details
    for all AutoML sessions associated with a given `dataset_id`.

    The results are ordered by session creation time, with the newest sessions first.
    """
    # 1. Check if the dataset exists using the CRUD operation
    db_dataset = crud_dataset.get(db, id=dataset_id)
    if not db_dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found."
        )

    # 2. Query for AutoML sessions linked to this dataset_id
    #    Order by creation date, newest first, or by ID as a consistent fallback.
    sessions_db = db.query(AutoMLSession)\
                    .filter(AutoMLSession.dataset_id == dataset_id)\
                    .first()

    if not sessions_db:
        return []

    return sessions_db

@router.get(
    "/datasets/{dataset_id}/finalized-models",
    response_model=list[FinalizedModelInfo],
    summary="List finalized models for a specific dataset",
    tags=["Finalized Models"]
)
def list_finalized_models_for_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves a list of all finalized models associated with the given dataset ID.
    This involves finding AutoML sessions linked to the dataset and then their
    corresponding finalized models.
    """
    # Check if dataset exists (optional, but good practice)
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {dataset_id} not found"
        )

    # Query to join FinalizedModel -> AutoMLSession and filter by dataset_id
    stmt = (
        select(FinalizedModel)
        .join(FinalizedModel.automl_sessions) # Use the relationship name
        .where(AutoMLSession.dataset_id == dataset_id)
        .options(
            # Eagerly load the related AutoMLSession to get its name easily
            # Use selectinload for one-to-many/many-to-many from parent perspective
            # Use joinedload for many-to-one/one-to-one from child perspective
            joinedload(FinalizedModel.automl_sessions)
        )
    )
    result = db.execute(stmt)
    finalized_models = result.scalars().unique().all() # .unique() needed due to join

    # Map to Pydantic schema, extracting session name
    response_data = []
    for model in finalized_models:
        response_data.append(
            FinalizedModelInfo(
                id=model.id,
                session_id=model.session_id,
                automl_session_name=model.automl_sessions.name if model.automl_sessions else "Unknown Session",
                model_name=model.model_name,
                created_at=model.created_at
            )
        )

    return response_data

@router.get(
    "/finalized-models/{finalized_model_id}/download",
    response_class=FileResponse, # Key part for file download
    summary="Download a specific finalized model (.pkl file)",
    tags=["Finalized Models"]
)
def download_finalized_model(
    finalized_model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Downloads the .pkl model file associated with the given finalized_model_id.
    """
    # Get the FinalizedModel record from the database
    model = db.get(FinalizedModel, finalized_model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"FinalizedModel with id {finalized_model_id} not found"
        )

    file_path = model.saved_model_path

    # --- Security and Existence Check ---
    # Basic check: Does the file exist?
    if not os.path.isfile(file_path):
        # Log this error server-side as it indicates an inconsistency
        print(f"ERROR: File not found for FinalizedModel {finalized_model_id}: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 404 if you prefer user not know why
            detail="Model file not found on server."
        )

    # Optional stronger check: Ensure the path is within an expected base directory
    # to prevent potential path traversal if save paths could be manipulated.
    # Example:
    # ALLOWED_BASE_PATH = "/path/to/your/safe/model/storage"
    # if not os.path.abspath(file_path).startswith(ALLOWED_BASE_PATH):
    #     print(f"SECURITY WARNING: Attempt to access file outside allowed path: {file_path}")
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Access to the specified file path is forbidden."
    #     )
    # --- End Security Check ---


    # Extract filename for the download prompt
    filename = os.path.basename(file_path)
    # Suggest a potentially more user-friendly download name (optional)
    suggested_filename = f"{model.model_name}_session{model.session_id}.pkl"

    return FileResponse(
        path=file_path,
        filename=suggested_filename, # Name suggested to the user for saving
        media_type='application/octet-stream' # Standard for binary file download
    )