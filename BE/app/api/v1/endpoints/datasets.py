from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query, status, Body
import io
from typing import List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
import pandas as pd
from app.crud.datasets import DatasetCreateSchema, DatasetUpdateSchema
from app.schemas.dataset import (
    DatasetFromConnection, 
    DatasetId, 
    DatasetPreview, 
    DatasetFlatListResponse, 
    DatasetAnalysisReport, 
    DatasetFlatInfo, 
    DataPreviewSchema, 
    DetailResponse,
    DatasetUpdateProjectName
)

# Import CRUD functions for datasets and connections
from app.crud.datasets import crud_dataset
from app.crud.connections import get_connection_by_id

# Import common dependencies and utilities
from app.api.v1.dependencies import get_db
from app.utils.file_storage import (
    save_dataframe_as_csv, get_file_path, load_csv_as_dataframe, save_dask_dataframe_as_csv, load_csv_as_dask_dataframe
)
from app import schemas
from ydata_profiling import ProfileReport
from fastapi.responses import FileResponse, HTMLResponse
import dask.dataframe as dd
import asyncio
import os


# --- API Router Setup ---
router = APIRouter(
    prefix='/v1',
    tags=['datasets']
)

# --- Endpoint Definitions ---

@router.post('/datasets/', response_model=DatasetId)
async def upload_datasets(project_name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    # --- Stage uploaded file to disk (for Dask to read efficiently) ---
    # This part is crucial for very large files to avoid memory issues
    # We'll create a temporary file path.
    # In a real app, use tempfile module for secure temporary files.
    temp_file_path = f"temp_{file.filename}" # Potentially add UUID for uniqueness
    try:
        with open(temp_file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # Read in 1MB chunks
                buffer.write(content)
        await file.close() # Ensure file is closed

        # Now use Dask to read the staged file
        try:
            # Run Dask's read_csv in a thread to avoid blocking asyncio event loop
            ddf = await asyncio.to_thread(dd.read_csv, temp_file_path, blocksize='64MB')
            # For basic validation or to get column names before saving, you might compute a small part:
            # await asyncio.to_thread(ddf.head)
        except Exception as e: # Catch Dask parsing errors
            raise HTTPException(status_code=400, detail=f"Error parsing CSV file with Dask: {e}")

        # 5. Create Initial Database Record (same as before)
        dataset_in_create = DatasetCreateSchema(project_name=project_name, connection_id=None)
        ds = None
        try:
            ds = crud_dataset.create(db=db, obj_in=dataset_in_create)
        except IntegrityError as e:
            db.rollback()
            error_detail = str(e.orig).lower()
            if "unique constraint" in error_detail and "project_name" in error_detail:
                raise HTTPException(status_code=409, detail=f"Project name '{project_name}' already exists.")
            else:
                raise HTTPException(status_code=500, detail=f"Database integrity error: {e}")
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating dataset record in DB: {e}")
        if not ds:
            raise HTTPException(status_code=500, detail="Failed to create dataset record, unknown error.")

        filename = f"dataset_{ds.id}.csv"
        filename_cleaned = f"dataset_{ds.id}_cleaned.csv" # You'll save this too

        saved_file_path = None
        try:
            # Use Dask's to_csv, run in a thread
            # save_dask_dataframe_as_csv uses ddf.to_csv which can be blocking
            saved_file_path = await asyncio.to_thread(save_dask_dataframe_as_csv, ddf, filename)
            await asyncio.to_thread(save_dask_dataframe_as_csv, ddf, filename_cleaned) # Save cleaned version too
        except Exception as e:
            crud_dataset.remove(db=db, id=ds.id)
            db.commit()
            raise HTTPException(status_code=500, detail=f"Failed to save Dask DataFrame: {e}")

        dataset_in_update = DatasetUpdateSchema(file_path=saved_file_path, project_name=ds.project_name)
        ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)
        return DatasetId(id=ds_updated.id)

    finally:
        # Clean up the temporary staged file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if file: # Ensure the UploadFile is closed if an error occurred before explicit close
            await file.close()

@router.post('/datasets/from-connection/', response_model=DatasetId)
async def ingest_from_connection(payload: DatasetFromConnection, db: Session = Depends(get_db)):
    """
    **Ingest Dataset from Database Connection via SQL Query**

    Initiates the process of creating a new dataset resource by executing a SQL query
    against a pre-configured database connection.

    This endpoint performs the following actions sequentially:
    1.  Validates the request body (`payload`) using Pydantic (`DatasetFromConnection`).
    2.  Retrieves the database connection details (`host`, `port`, `user`, etc.) using the `payload.connection_id` via `get_connection_by_id`.
    3.  Validates that the connection details were found.
    4.  Constructs a SQLAlchemy database connection URL string from the retrieved details.
    5.  Creates a temporary SQLAlchemy engine using `create_engine`.
    6.  Connects to the target database using the engine.
    7.  Executes the SQL query provided in `payload.query` using `pd.read_sql` to fetch data into a pandas DataFrame.
    8.  Disposes of the SQLAlchemy engine to release resources (`engine.dispose()`).
    9.  Creates an initial `Dataset` record in the database, linking it to the `payload.connection_id`, via `crud_dataset.create`.
    10. Constructs a filename for storage using the newly generated dataset ID (e.g., `dataset_{id}.csv`).
    11. Saves the fetched pandas DataFrame to the configured file storage as a CSV file using `save_dataframe_as_csv`.
        *   If file saving fails, the previously created database record (Step 9) is deleted (rollback).
    12. Updates the `Dataset` record in the database with the actual path to the saved file via `crud_dataset.update`.
    13. Returns the unique ID (`id`) of the successfully created dataset record.

    Args:
        `payload` (DatasetFromConnection): Request body containing:
            - `connection_id` (int): ID of the pre-configured database connection.
            - `query` (str): The SQL query to execute.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - `404`: If the `connection_id` is not found.
            - `500`: If constructing the database URL fails (e.g., missing connection details).
            - `400`: If connecting to the database or executing the SQL query fails.
            - `500`: If saving the fetched DataFrame to file storage fails after initial DB record creation.

    Returns:
        `DatasetId`: An object containing the assigned `id` for the new dataset.
    """
    # 1. Validation (Implicit by Pydantic)

    # 2. Retrieve Connection Details
    conn = get_connection_by_id(db, payload.connection_id)
    # 3. Validate connection exists
    if not conn:
        raise HTTPException(status_code=404, detail='Connection not found')

    # 4. Construct Database URL
    try:
        db_url = (
            f'{conn.type}://{conn.username}:{conn.password}@'
            f'{conn.host}:{conn.port}/{conn.database}'
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error building database URL: {e}")

    # 5, 6, 7, 8. Execute Query and Fetch Data (with resource disposal)
    engine = None # Ensure engine is defined for finally block
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
             df = pd.read_sql(payload.query, con=connection)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to execute query or read data: {e}')
    finally:
        if engine:
            engine.dispose() # Release connection pool resources

    # 9. Create initial DB record
    dataset_in_create = DatasetCreateSchema(connection_id=payload.connection_id)
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 10. Construct filename
    filename = f"dataset_{ds.id}.csv"
    # 11. Save the dataframe (with rollback)
    try:
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # Cleanup DB record if saving fails
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 12. Update the DB record with the path
    dataset_in_update = DatasetUpdateSchema(file_path=path)
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    # 13. Return the Dataset ID
    return DatasetId(id=ds_updated.id)

@router.get('/datasets/{dataset_id}/preview', response_model=DatasetPreview)
async def get_dataset_preview(dataset_id: int, db: Session = Depends(get_db)): # Make async
    file_path = get_file_path(f'dataset_{dataset_id}_cleaned.csv')

    try:
        # Load as Dask DataFrame, run in thread
        ddf = await asyncio.to_thread(load_csv_as_dask_dataframe, file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Cleaned dataset file not found.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Get total rows and columns from Dask DataFrame (efficient)
    total_row = await asyncio.to_thread(lambda: len(ddf)) # len(ddf) computes if not already known
    total_col = len(ddf.columns)
    
    preview_row_count = min(100, total_row)
    if preview_row_count == 0 and total_row > 0: # if total_row is >0 but preview_row_count became 0
        preview_row_count = total_row # ensure at least one row if dataset not empty

    if total_row == 0 : # Handle empty dataframe
        preview_df = pd.DataFrame(columns=ddf.columns)
    else:
        # Get the head and compute it to a Pandas DataFrame, run in thread
        preview_df = await asyncio.to_thread(ddf.head, preview_row_count) # ddf.head() is lazy, .compute() is implicit by some functions or explicit

    project_name_obj = crud_dataset.get(db, id=dataset_id)
    if not project_name_obj:
        raise HTTPException(status_code=404, detail="Dataset metadata not found.")
    project_name = project_name_obj.project_name

    try:
        df_for_schema = preview_df.reset_index(drop=True)
        # Pandas .astype(object).where(pd.notnull) can be slow on very wide previews.
        # For a 100-row preview, it's usually acceptable.
        data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
        preview_schema = schemas.DataFrameStructure(
            columns=df_for_schema.columns.tolist(),
            data=data_list
        )
    except Exception as convert_e:
        print(f"Error converting preview DataFrame to schema: {convert_e}")
        raise HTTPException(status_code=500, detail="Failed to format dataset preview data.")

    response_data = DatasetPreview(
        preview_data=preview_schema,
        preview_row=len(preview_df), # Use actual length of preview_df
        total_row=total_row,
        total_col=total_col,
        project_name=project_name
    )
    return response_data

@router.get(
    "/datasets/all-by-creation/",
    response_model=DatasetFlatListResponse,
    summary="List ALL Datasets Ordered by Creation Date (Newest First)"
)
async def list_all_datasets_ordered_by_creation_date( # Make async
    db: Session = Depends(get_db)
):
    db_dataset_models = crud_dataset.get_all_datasets_ordered_by_creation(db=db, descending=True)
    response_datasets: List[DatasetFlatInfo] = []

    for dataset_model in db_dataset_models:
        data_preview_payload: Optional[DataPreviewSchema] = None
        ddf: Optional[dd.DataFrame] = None

        model_id = getattr(dataset_model, 'id', None)
        original_file_path = getattr(dataset_model, 'file_path', None)
        
        # Determine which file to load (cleaned or original)
        path_to_load = None
        if model_id is not None:
            try:
                cleaned_filename_candidate = f"dataset_{model_id}_cleaned.csv"
                potential_cleaned_path = get_file_path(cleaned_filename_candidate)
                # Check existence before trying to load
                if os.path.exists(potential_cleaned_path):
                    path_to_load = potential_cleaned_path
                elif original_file_path and isinstance(original_file_path, str) and os.path.exists(original_file_path):
                    path_to_load = original_file_path
            except Exception: # Broad catch if get_file_path or os.path.exists fails
                 if original_file_path and isinstance(original_file_path, str) and os.path.exists(original_file_path):
                    path_to_load = original_file_path
        elif original_file_path and isinstance(original_file_path, str) and os.path.exists(original_file_path):
            path_to_load = original_file_path

        if path_to_load:
            try:
                # Load with Dask, run in thread
                current_ddf = await asyncio.to_thread(load_csv_as_dask_dataframe, path_to_load, blocksize='16MB') # Smaller blocksize for just a head
                
                # Get a small preview
                num_rows_to_take = min(4, await asyncio.to_thread(lambda: len(current_ddf)))
                num_cols_to_take = min(4, len(current_ddf.columns))

                if num_rows_to_take > 0 and num_cols_to_take > 0:
                    # Select columns first (Dask is lazy) then head, then compute
                    cols_for_preview = current_ddf.columns[:num_cols_to_take].tolist()
                    preview_df_slice = await asyncio.to_thread(current_ddf[cols_for_preview].head, num_rows_to_take)
                else:
                    preview_df_slice = pd.DataFrame() # Empty df

                if not preview_df_slice.empty:
                    column_names = [str(col) for col in preview_df_slice.columns.tolist()]
                    serializable_data_rows: List[List[Any]] = []
                    for _, row_series in preview_df_slice.iterrows():
                        processed_row: List[Any] = []
                        for item in row_series:
                            if pd.isna(item):
                                processed_row.append(None)
                            elif isinstance(item, (datetime, pd.Timestamp)):
                                processed_row.append(item.isoformat())
                            else:
                                processed_row.append(item)
                        serializable_data_rows.append(processed_row)
                    data_preview_payload = DataPreviewSchema(
                        columns=column_names,
                        data=serializable_data_rows
                    )
            except (FileNotFoundError, RuntimeError, Exception) as e_load_preview:
                # print(f"Could not load/preview dataset {model_id}: {e_load_preview}")
                data_preview_payload = None # Keep it None if any error

        dataset_flat_info = DatasetFlatInfo(
            id=dataset_model.id,
            project_name=getattr(dataset_model, 'project_name', None),
            created_at=dataset_model.created_at,
            is_model=getattr(dataset_model, 'is_model', False),
            is_clean=getattr(dataset_model, 'is_clean', False),
            data_preview=data_preview_payload
        )
        response_datasets.append(dataset_flat_info)
    
    return DatasetFlatListResponse(datasets=response_datasets)

@router.get(
    "/datasets/{dataset_id}/analysis-report/",
    response_model=DatasetAnalysisReport,
    summary="Get Detailed Analysis Report for a Dataset"
)
async def get_dataset_analysis_report(dataset_id: int, db: Session = Depends(get_db)): # Make async
    try:
        # Assume you have a dask-enabled version in your crud, or adapt crud_dataset.get_dataset_analysis
        # For this example, let's imagine it's refactored as get_dataset_analysis_dask
        report = await crud_dataset.get_dataset_analysis_dask(db=db, dataset_id=dataset_id) # This needs to be async and use Dask
        if report is None:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found or analysis failed.")
        return report
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) # More specific message if possible
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error during dataset analysis for ID {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the dataset.")

    
@router.delete(
    "/datasets/{dataset_id}",
    response_model=DetailResponse, # Response is a simple message
    status_code=status.HTTP_200_OK, # Or 204 if you prefer no content
    summary="Delete a dataset",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Dataset not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during deletion"},
        # Add others if applicable (e.g., 403 Forbidden)
    }
)
def delete_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int
) -> DetailResponse:
    """
    Deletes a dataset specified by its ID.

    This will also delete associated records if cascade options are set
    (e.g., CleaningJob, AutoMLSession, ChatSessionState).
    """

    # Use the remove method from CRUDBase via crud_dataset instance
    deleted_dataset = crud_dataset.remove(db=db, id=dataset_id)

    if not deleted_dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {dataset_id} not found."
        )

    return DetailResponse(detail=f"Dataset with id {dataset_id} deleted successfully.")

@router.patch(
    "/datasets/{dataset_id}/project_name", # More specific path for updating just the name
    response_model=DatasetFlatInfo, # Return updated dataset info
    status_code=status.HTTP_200_OK,
    summary="Update dataset project name",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Dataset not found"},
        status.HTTP_409_CONFLICT: {"description": "Project name already exists"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error (e.g., empty string)"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during update"},
    }
)
def update_dataset_project_name(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    update_data: DatasetUpdateProjectName = Body(...) # Get data from request body
) -> DatasetFlatInfo:
    """
    Updates the 'project_name' for a specific dataset.
    The new name must be unique across all datasets if it is not null.
    """

    # Use the specific CRUD method
    updated_dataset, error_msg = crud_dataset.update_project_name(
        db=db,
        dataset_id=dataset_id,
        new_project_name=update_data.project_name
    )

    if error_msg:
        # Check if the dataset itself was found but validation failed
        if updated_dataset is not None: # Dataset exists, but name conflict or DB error
             if "already exists" in error_msg:
                 status_code = status.HTTP_409_CONFLICT
             else: # Other database error during commit
                 status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
             raise HTTPException(status_code=status_code, detail=error_msg)
        else:
              raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inconsistent server state during update.")


    if updated_dataset is None:
        # This means the initial get in update_project_name returned None -> dataset not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {dataset_id} not found."
        )

    return updated_dataset

@router.get(
    "/datasets/{dataset_id}/profile/download",
    summary="Generate and download ydata-profiling report",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/html": {}},
            "description": "Returns the ydata-profiling report as an HTML file attachment.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Dataset not found or source file missing"},
        status.HTTP_409_CONFLICT: {"description": "Dataset found but has no associated file path"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Error loading data or generating profile"},
    }
)
async def download_dataset_profile( # Make async
    *,
    db: Session = Depends(get_db),
    dataset_id: int
) -> HTMLResponse:
    db_dataset = crud_dataset.get(db=db, id=dataset_id) # This can remain synchronous
    if not db_dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {dataset_id} not found.")
    if not db_dataset.file_path or not os.path.exists(db_dataset.file_path): # Check existence
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source file for dataset {dataset_id} not found or path missing.")

    try:
        # Load with Dask in a thread
        ddf = await asyncio.to_thread(load_csv_as_dask_dataframe, db_dataset.file_path)
        
        # Check if ddf is empty after loading
        # len(ddf) will trigger a compute if the length isn't known, so do it carefully
        is_empty = await asyncio.to_thread(lambda: ddf.npartitions == 0 or len(ddf.columns) == 0 or len(ddf) == 0)
        if is_empty:
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"The dataset file for id {dataset_id} is empty or could not be loaded properly."
             )

    except FileNotFoundError: # Should be caught by os.path.exists above, but good fallback
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source data file not found for dataset id {dataset_id}.")
    except RuntimeError as e: # From load_csv_as_dask_dataframe
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load data for dataset {dataset_id}: {e}")
    except Exception as e: # Other general errors
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load data for dataset {dataset_id}: {e}")


    # Generate Profile Report using Dask DataFrame - this is CPU intensive!
    def generate_profile_blocking(dask_df, title):
        # ydata-profiling can work directly with Dask DataFrames.
        # It will internally trigger computations as needed.
        profile = ProfileReport(
            dask_df.compute(),
            title=title,
            minimal=True
        )
        return profile.to_html()

    try:
        html_report_content = await asyncio.to_thread(
            generate_profile_blocking,
            ddf,
            f"Dataset Profile: {db_dataset.project_name or f'ID {dataset_id}'}"
        )
    except Exception as e:
        # Log the full error: import logging; logging.exception("Profile generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate profile report for dataset id {dataset_id}: {type(e).__name__} - {e}"
        )

    file_name = f"dataset_{dataset_id}_profile.html"
    headers = {"Content-Disposition": f"attachment; filename=\"{file_name}\""}
    return HTMLResponse(content=html_report_content, status_code=status.HTTP_200_OK, headers=headers, media_type="text/html")