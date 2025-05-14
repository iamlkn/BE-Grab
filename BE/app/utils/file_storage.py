import os
from pathlib import Path
import pandas as pd
import dask.dataframe as dd # New import for Dask
from dotenv import load_dotenv

load_dotenv()

# Base directory for storing CSVs/Datasets
STORAGE_DIR = Path(os.getenv('DATA_STORAGE_DIR', 'data'))
# Create storage dir once at module load

# --- Pandas based functions (for smaller data or specific Pandas operations) ---

def save_dataframe_as_csv(df: pd.DataFrame, filename: str) -> str:
    """Saves a Pandas DataFrame to a CSV file in the STORAGE_DIR."""
    path = STORAGE_DIR / filename
    df.to_csv(path, index=False)
    return str(path)

def load_csv_as_dataframe(file_path_str: str) -> pd.DataFrame:
    """Loads a CSV file from the given path into a Pandas DataFrame."""
    path = Path(file_path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {file_path_str}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path_str}")
    return pd.read_csv(path)

# --- Dask based functions (for larger datasets) ---

def save_dask_dataframe_as_csv(ddf: dd.DataFrame, filename: str) -> str:
    """
    Saves a Dask DataFrame to a single CSV file in the STORAGE_DIR.
    
    Note: For very large Dask DataFrames, `single_file=True` can be a bottleneck
    as it might consolidate all data on a single worker before writing.
    Consider `ddf.to_csv('directory_path/', ...)` to write part-files if performance
    is critical and you can manage multiple output files, or
    `ddf.compute().to_csv(path, index=False)` if the computed Pandas DataFrame fits in memory.
    """
    path = STORAGE_DIR / filename
    ddf.to_csv(str(path), single_file=True, index=False) # dask to_csv usually expects string path
    return str(path)

def load_csv_as_dask_dataframe(file_path_str: str, blocksize: str = '64MB') -> dd.DataFrame:
    """
    Loads a CSV file from the given path into a Dask DataFrame.
    
    Args:
        file_path_str (str): The full path to the CSV file.
        blocksize (str): Size of chunks for Dask to read (e.g., '64MB', '128MB').
    """
    path = Path(file_path_str)
    if not path.exists():
        raise FileNotFoundError(f"Dask: File not found at {file_path_str}")
    if not path.is_file():
        raise ValueError(f"Dask: Path is not a file: {file_path_str}")
    
    return dd.read_csv(str(path), blocksize=blocksize)

# --- Path Utilities ---

def get_file_path(filename: str) -> str:
    """Constructs the full path for a given filename within the STORAGE_DIR."""
    path = STORAGE_DIR / filename
    return str(path)

def get_cleaned_df_path(raw_file_path_str: str) -> str:
    """
    Gets the path for the '_cleaned' version of a raw data file.
    If the cleaned file doesn't exist, it creates it by copying the raw file
    using Dask (efficient for large files).
    
    Args:
        raw_file_path_str (str): Path to the original/raw data file.
        
    Returns:
        str: Path to the cleaned data file.
    """
    raw_file_path = Path(raw_file_path_str)
    if not raw_file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file_path_str}")
    if not raw_file_path.is_file():
        raise ValueError(f"Raw path is not a file: {raw_file_path_str}")

    # Construct the name and path for the cleaned file
    cleaned_name = raw_file_path.stem + "_cleaned" + raw_file_path.suffix
    cleaned_file_path = raw_file_path.with_name(cleaned_name) 

    if not cleaned_file_path.exists():
        print(f"Cleaned file '{cleaned_file_path}' not found. Creating from '{raw_file_path_str}' using Dask.")
        try:
            # Load raw file with Dask
            ddf_raw = load_csv_as_dask_dataframe(raw_file_path_str)
            
            # Save it as the cleaned version using Dask.
            save_dask_dataframe_as_csv(ddf_raw, cleaned_name)
            print(f"Successfully created cleaned file: {cleaned_file_path}")
        except Exception as e:
            print(f"Error creating cleaned file {cleaned_file_path} from {raw_file_path_str}: {e}")
            # Depending on desired behavior, you might re-raise or handle differently
            raise RuntimeError(f"Failed to create cleaned file {cleaned_file_path}: {e}") from e
            
    return str(cleaned_file_path)