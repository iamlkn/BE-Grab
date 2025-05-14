from fastapi import APIRouter, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import numpy as np
import asyncio # For async operations
from functools import lru_cache # Simple lru_cache for schema
from cachetools import LRUCache # More advanced LRU cache for DataFrames


router = APIRouter(
    prefix='/v1',
    tags=['chart']
)

# --- Cache Configuration ---
# Cache for inferred schemas (column names, logical types, pandas dtypes)
# Stores tuples: (columns_list, logical_types_dict, pandas_dtypes_dict)
SCHEMA_CACHE_MAX_SIZE = 50
schema_cache = LRUCache(maxsize=SCHEMA_CACHE_MAX_SIZE)

# Cache for loaded DataFrames
# Using LRUCache to limit memory usage. Maxsize is number of DataFrames.
# Adjust maxsize based on typical DataFrame sizes and available memory.
DATAFRAME_CACHE_MAX_SIZE = 10
csv_cache = LRUCache(maxsize=DATAFRAME_CACHE_MAX_SIZE)

# Number of rows to read for schema inference
SCHEMA_INFERENCE_ROWS = 1000

# --- Helper Functions ---
from app.utils.file_storage import get_file_path

async def _infer_datetime_series(series: pd.Series, sample_size: int = 100) -> pd.Series:
    """
    Asynchronously tries to convert a series to datetime based on common formats.
    Operates on a sample.
    """
    # Take a sample of non-NA string values
    str_values = series.dropna().astype(str)
    if str_values.empty:
        return series # Return original if no data to infer from

    sample_values = str_values.head(sample_size)
    if sample_values.empty: # if original series had values but all were NA
        return series

    best_fmt = None
    max_parsed_ratio = 0.0

    # Test common datetime formats
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                '%d-%m-%Y', '%m-%d-%Y']:
        try:
            # Use asyncio.to_thread for the potentially blocking pd.to_datetime call
            parsed_sample = await asyncio.to_thread(
                pd.to_datetime, sample_values, format=fmt, errors='coerce'
            )
            parsed_ratio = parsed_sample.notna().mean()
            if parsed_ratio > max_parsed_ratio:
                max_parsed_ratio = parsed_ratio
                best_fmt = fmt
        except Exception:
            continue
    
    # If a good format is found (>70% of sample parsed), convert the whole series
    if best_fmt and max_parsed_ratio > 0.7:
        return await asyncio.to_thread(pd.to_datetime, series, format=best_fmt, errors='coerce')
    return series


async def infer_schema_from_csv(table_name: str):
    """
    Infers schema (column names, logical types, pandas dtypes) by reading a sample of the CSV.
    Caches the result.
    """
    if table_name in schema_cache:
        return schema_cache[table_name]

    try:
        file_path = get_file_path(table_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file for table '{table_name}' not found at {file_path}")

        # Use asyncio.to_thread for the blocking pd.read_csv call
        df_sample = await asyncio.to_thread(pd.read_csv, file_path, nrows=SCHEMA_INFERENCE_ROWS)
        df_sample.columns = [col.strip()for col in df_sample.columns]

        logical_types = {}
        pandas_dtypes = {}

        for col in df_sample.columns:
            original_series = df_sample[col]
            # Attempt datetime conversion first on the sample
            series_after_dt_check = await _infer_datetime_series(original_series.copy(), sample_size=min(100, SCHEMA_INFERENCE_ROWS))
            
            if pd.api.types.is_datetime64_any_dtype(series_after_dt_check):
                logical_types[col] = "datetime"
                pandas_dtypes[col] = 'datetime64[ns]' # Be explicit for read_csv
            elif pd.api.types.is_numeric_dtype(series_after_dt_check): # Check numeric after datetime
                logical_types[col] = "number"
                # Keep original numeric dtype (int64, float64)
                pandas_dtypes[col] = series_after_dt_check.dtype
            elif pd.api.types.is_bool_dtype(series_after_dt_check):
                logical_types[col] = "boolean"
                pandas_dtypes[col] = 'bool'
            else: # Default to string
                # Check for low cardinality strings to suggest 'category' dtype for memory saving
                if original_series.nunique() / len(original_series) < 0.5 and len(original_series) > 100: # Heuristic
                    logical_types[col] = "category" # or "string_low_cardinality"
                    pandas_dtypes[col] = "category"
                else:
                    logical_types[col] = "string"
                    pandas_dtypes[col] = "object" # or pd.StringDtype() for newer pandas

        schema_info = (df_sample.columns.tolist(), logical_types, pandas_dtypes)
        schema_cache[table_name] = schema_info
        return schema_info

    except FileNotFoundError:
        raise
    except Exception as e:
        # Log the error e
        raise RuntimeError(f"Error inferring schema for '{table_name}': {str(e)}")


async def load_csv_data(dataset_id: str, columns_to_load: list[str] = None) -> pd.DataFrame:
    """
    Loads CSV data, potentially using cached schema for optimized dtype loading.
    Can load specific columns. Cache key includes columns_to_load.
    """
    table_name = f'dataset_{dataset_id}_cleaned.csv'
    cache_key = (table_name, tuple(sorted(columns_to_load)) if columns_to_load else None)
    if cache_key in csv_cache:
        return csv_cache[cache_key].copy() # Return a copy to prevent mutation of cached df

    try:
        file_path = get_file_path(table_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file for table '{table_name}' not found at {file_path}")

        # Get schema to use for dtype hints
        _, _, pandas_dtypes_map = await infer_schema_from_csv(table_name)
        
        # Prepare dtypes for pd.read_csv. Only use dtypes for columns we are loading.
        # effective_dtypes = {
        #     col: dtype 
        #     for col, dtype in pandas_dtypes_map.items() 
        #     if columns_to_load is None or col in columns_to_load
        # }
        
        parse_date_cols = []
        effective_dtypes = {}

        for col, dtype in pandas_dtypes_map.items():
            if columns_to_load is not None and col not in columns_to_load:
                continue
            if dtype == "datetime64[ns]":
                parse_date_cols.append(col)
            else:
                effective_dtypes[col] = dtype

        # Use asyncio.to_thread for the blocking pd.read_csv call
        df = await asyncio.to_thread(
            pd.read_csv,
            file_path,
            usecols=columns_to_load,
            dtype=effective_dtypes,
            parse_dates=parse_date_cols
            # na_filter=True, # default
            # low_memory=False # May help with mixed types, but dtype spec should handle
        )
        # Ensure column names match the sanitized ones from schema inference
        df.columns = [col.strip() for col in df.columns]

        # Perform full series datetime conversion if schema suggested it, but read_csv didn't fully parse
        # This is a fallback, as `dtype` in read_csv should ideally handle it.
        for col in df.columns:
            if pandas_dtypes_map.get(col) == 'datetime64[ns]' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = await _infer_datetime_series(df[col]) # Re-run full inference on loaded column

        csv_cache[cache_key] = df
        return df.copy() # Return a copy

    except FileNotFoundError:
        raise
    except Exception as e:
        # Log the error e
        raise RuntimeError(f"Error loading data for '{table_name}': {str(e)}")

# --- API Endpoints ---

# @router.get("/load_table_schema/")
# async def load_table_schema_endpoint(dataset_id: int = Query(...)):
#     """Loads/infers and caches the schema for a table."""
#     try:
#         table = f'dataset_{dataset_id}_cleaned.csv'
#         await infer_schema_from_csv(table)
#         return {"message": f"Schema for table '{table}' processed and cached."}
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except RuntimeError as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     except ValueError as e: # For invalid table name
#         raise HTTPException(status_code=400, detail=str(e))


@router.get("/get_columns/")
async def get_columns(dataset_id: int = Query(...)):
    """Gets column names and their inferred logical types for a table."""
    try:
        table = f'dataset_{dataset_id}_cleaned.csv'
        columns, logical_types, _ = await infer_schema_from_csv(table)
        return {
            "table_name": table,
            "columns": [{"name": col, "logical_type": logical_types.get(col, "unknown")} for col in columns]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e: # For invalid table name
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get_chart_columns/")
async def get_chart_columns(dataset_id: int = Query(...), chart_type: str = Query(...)):
    """Suggests columns suitable for X and Y axes based on chart type and inferred column types."""
    try:
        table = f'dataset_{dataset_id}_cleaned.csv'
        columns, logical_types, _ = await infer_schema_from_csv(table)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e: # For invalid table name
        raise HTTPException(status_code=400, detail=str(e))

    column_details = [{"name": col, "logical_type": logical_types.get(col, "string")} for col in columns]

    if chart_type == "bar":
        valid_x_types = ["string", "datetime", "category"]
        valid_y_types = ["number"]
    elif chart_type == "line":
        valid_x_types = ["number", "datetime"]
        valid_y_types = ["number"]
    elif chart_type == "scatter":
        valid_x_types = ["number", "datetime"] # Datetime can be X for scatter
        valid_y_types = ["number"]
    elif chart_type == "histogram":
        valid_x_types = ["number"]
        valid_y_types = []  # No explicit Y column needed from user
    else:
        raise HTTPException(status_code=400, detail="Unsupported chart type")

    x_columns = [col["name"] for col in column_details if col["logical_type"] in valid_x_types]
    y_columns = [col["name"] for col in column_details if col["logical_type"] in valid_y_types]

    return {
        "x_columns": x_columns,
        "y_columns": y_columns if chart_type != "histogram" else []
    }


# Grouping and aggregation can be CPU-bound, run in a thread
def _perform_aggregation(df, x_col, y_col, x_type):
    if x_type == "datetime":

            summary = df.groupby(x_col)[y_col].mean().reset_index().sort_values(by=x_col)
    elif x_type == "category" or x_type == "string":
        # For categorical/string X, might want to limit top N categories if too many
        # For simplicity, group by all.
        summary = df.groupby(x_col)[y_col].mean().reset_index()
        # Sort by y value for bar charts often makes sense, or by x_col alphabetically
        summary = summary.sort_values(by=y_col, ascending=False) # Example: sort by value
    else: # number
        summary = df.groupby(x_col)[y_col].mean().reset_index()
        if pd.api.types.is_numeric_dtype(summary[x_col]): # Sort numeric x-axis
            summary = summary.sort_values(by=x_col)


    # Convert x-axis to string for JSON serialization, esp. for datetimes/categories
    summary[x_col] = summary[x_col].astype(str)
    return summary

@router.get("/get_summary/")
async def get_summary(dataset_id: int = Query(...), x_column: str = Query(...), y_column: str = Query(None), bins: int = Query(10)):
    """
    Generates summary data for charts.
    For histograms (y_column is None), uses the x_column (must be numeric).
    For other charts, aggregates y_column by x_column.
    """
    try:
        table = f'dataset_{dataset_id}_cleaned.csv'
        # Get schema to validate columns and get their types
        all_columns, logical_types_map, pandas_dtypes_map = await infer_schema_from_csv(table)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e: # For invalid table name
        raise HTTPException(status_code=400, detail=str(e))


    if x_column not in all_columns:
        raise HTTPException(status_code=400, detail=f"Invalid x_column: '{x_column}' not found in table '{table}'.")

    x_logical_type = logical_types_map.get(x_column)

    # Histogram case (only X column)
    if y_column is None:
        if x_logical_type != "number":
            raise HTTPException(status_code=400, detail=f"For histogram, x_column ('{x_column}') must be numeric, but it's '{x_logical_type}'.")
        
        # Load only the necessary x_column
        df_x = await load_csv_data(dataset_id, columns_to_load=[x_column])
        df_x = df_x[[x_column]].dropna()

        if df_x.empty:
             return {"x": [], "y": []} # Or appropriate empty response

        # Use asyncio.to_thread for numpy.histogram
        hist, bin_edges = await asyncio.to_thread(np.histogram, df_x[x_column], bins=bins)
        
        x_labels = [f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}" for i in range(len(hist))]
        return {
            "x_axis_label": x_column,
            "y_axis_label": "Frequency",
            "x": x_labels,
            "y": hist.tolist()
        }
        
    # Aggregation case (X and Y columns)
    if y_column not in all_columns:
        raise HTTPException(status_code=400, detail=f"Invalid y_column: '{y_column}' not found in table '{table}'.")

    y_logical_type = logical_types_map.get(y_column)
    if y_logical_type != "number":
        raise HTTPException(status_code=400, detail=f"For aggregation, y_column ('{y_column}') must be numeric, but it's '{y_logical_type}'.")

    # Load only x_column and y_column
    df_xy = await load_csv_data(dataset_id, columns_to_load=[x_column, y_column])
    df_xy = df_xy[[x_column, y_column]].dropna() # Drop rows where EITHER x or y is NA for aggregation

    if df_xy.empty:
        return {"x": [], "y": []}

    summary_df = await asyncio.to_thread(_perform_aggregation, df_xy, x_column, y_column, x_logical_type)

    return {
        "x_axis_label": x_column,
        "y_axis_label": f"Average of {y_column}",
        "x": summary_df[x_column].tolist(),
        "y": summary_df[y_column].tolist()
    }