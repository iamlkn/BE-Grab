from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

# Import database models and dependencies
from app.api.v1.dependencies import get_db
from app.db.models.datasets import Dataset
from app.utils.file_storage import load_csv_as_dataframe

# --- API Router Setup ---
router = APIRouter(
    prefix='/v1/datasets',
    tags=['EDA']
)

# --- Endpoint Definitions ---

@router.get('/{dataset_id}/eda/stats')
def get_summary_statistics(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Summary Statistics for a Dataset**

    Calculates and returns descriptive statistics for all columns in the specified dataset.

    This endpoint performs the following actions sequentially:
    1.  Queries the database to find the `Dataset` record corresponding to the provided `dataset_id`.
    2.  Validates that a dataset record was found. If not, raises a 404 error.
    3.  Retrieves the `file_path` from the found dataset record.
    4.  Loads the dataset from the CSV file specified by `file_path` into a pandas DataFrame using `load_csv_as_dataframe`.
    5.  Calculates descriptive statistics for all columns (numeric and categorical) using `df.describe(include='all')`.
    6.  Replaces any `NaN` values in the resulting statistics DataFrame with empty strings (`''`) using `.fillna('')`.
    7.  Converts the statistics DataFrame into a dictionary format suitable for JSON response using `.to_dict()`.
    8.  Returns the dictionary containing the summary statistics.

    Args:
        `dataset_id` (int): The unique identifier of the dataset.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - `404`: If no dataset record is found for the given `dataset_id`.
            - Potentially (from `load_csv_as_dataframe`): If the dataset file cannot be found or loaded.

    Returns:
        `dict`: A dictionary representing the summary statistics, with NaN values replaced by empty strings.
    """
    # 1. Retrieve Dataset Record
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # 2. Validate Dataset Existence
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')

    # 3. Get file path (implicit in dataset object)
    # 4. Load Dataset Data
    df = load_csv_as_dataframe(dataset.file_path)

    # 5. Calculate Summary Statistics
    stats_df = df.describe(include='all')

    # 6. Format NaN values
    stats_df_filled = stats_df.fillna('')

    # 7. Convert to Dictionary
    stats_dict = stats_df_filled.to_dict()

    # 8. Return Dictionary
    return stats_dict


@router.get('/{dataset_id}/eda/corr')
def get_correlation_matrix(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Correlation Matrix for Numeric Columns in a Dataset**

    Calculates and returns the pairwise correlation matrix for all numeric columns
    in the specified dataset.

    This endpoint performs the following actions sequentially:
    1.  Queries the database to find the `Dataset` record corresponding to the provided `dataset_id`.
    2.  Validates that a dataset record was found. If not, raises a 404 error.
    3.  Retrieves the `file_path` from the found dataset record.
    4.  Loads the dataset from the CSV file specified by `file_path` into a pandas DataFrame using `load_csv_as_dataframe`.
    5.  Selects only the columns with numeric data types from the DataFrame using `df.select_dtypes(include='number')`.
    6.  Calculates the pairwise correlation matrix (default: Pearson) for the numeric columns using `.corr()`.
    7.  Replaces any `NaN` values in the resulting correlation matrix with `0` using `.fillna(0)`. (NaNs can occur for columns with zero variance).
    8.  Converts the correlation matrix DataFrame into a dictionary format suitable for JSON response using `.to_dict()`.
    9.  Returns the dictionary containing the correlation matrix.

    Args:
        `dataset_id` (int): The unique identifier of the dataset.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - `404`: If no dataset record is found for the given `dataset_id`.
            - Potentially (from `load_csv_as_dataframe`): If the dataset file cannot be found or loaded.

    Returns:
        `dict`: A dictionary representing the correlation matrix for numeric columns, with NaN values replaced by 0.
    """
    # 1. Retrieve Dataset Record
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # 2. Validate Dataset Existence
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found') # Corrected 'details' typo

    # 3. Get file path (implicit)
    # 4. Load Dataset Data
    df = load_csv_as_dataframe(dataset.file_path)

    # 5. Select Numeric Columns
    numeric_df = df.select_dtypes(include='number')

    # 6. Calculate Correlation Matrix
    corr_matrix = numeric_df.corr()

    # 7. Handle NaN Values
    corr_matrix_filled = corr_matrix.fillna(0)

    # 8. Convert to Dictionary
    corr_dict = corr_matrix_filled.to_dict()

    # 9. Return Dictionary
    return corr_dict