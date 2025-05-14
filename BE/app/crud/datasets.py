from sqlalchemy.orm import Session
from typing import Optional, List, Tuple
from pydantic import BaseModel # Assuming FeatureProfile uses Field for validation
# from pydantic import Field # If FeatureProfile uses Field for validation, e.g. Field(..., le=100.0)
from datetime import datetime
import pandas as pd

from app.db.models.datasets import Dataset
from app.crud.base import CRUDBase
from app.schemas.dataset import DatasetInfo, DatasetAnalysisReport, FeatureProfile # FeatureProfile is defined here
from app.utils.file_storage import get_file_path, load_csv_as_dataframe, load_csv_as_dask_dataframe, save_dask_dataframe_as_csv
# from ydata_profiling import ProfileReport # Not used in the provided snippet that causes error
import dask.dataframe as dd
import asyncio
import dask


def map_dtype_to_simplified_type(dtype_obj, series: pd.Series) -> str:
    """
    Maps a pandas dtype object to a simplified type string.
    For object types, it checks if it can be inferred as datetime.
    """
    if pd.api.types.is_integer_dtype(dtype_obj):
        return "integer"
    elif pd.api.types.is_float_dtype(dtype_obj):
        return "float"
    elif pd.api.types.is_datetime64_any_dtype(dtype_obj):
        return "datetime"
    elif pd.api.types.is_bool_dtype(dtype_obj):
        return "boolean"
    elif pd.api.types.is_object_dtype(dtype_obj):
        # Try to infer if object column is actually datetime-like
        try:
            # Attempt to convert a small sample to datetime to check
            # Be careful with large datasets, sample or use a more robust check
            if len(series) > 0:
                # Ensure series used for pd.to_datetime is a Pandas Series if series is Dask
                # However, this function seems intended for Pandas series input.
                # If 'series' can be a Dask series here, this sample needs dask computation.
                # Based on context, it's more likely used outside the Dask pipeline or on computed Dask series.
                pd.to_datetime(series.dropna().sample(min(5, len(series.dropna()))), errors='raise')
                return "datetime" # If conversion works for a sample, assume datetime
            return "categorical" # Default for object
        except (ValueError, TypeError):
            return "categorical" # If conversion fails, it's likely string/categorical
    else:
        return str(dtype_obj) # Fallback to the original dtype string

# --- Schemas ---
class DatasetCreateSchema(BaseModel):
    project_name: str
    file_path: str = ""
    connection_id: Optional[int] = None

class DatasetUpdateSchema(BaseModel):
    file_path: Optional[str] = None
    connection_id: Optional[int] = None

# --- CRUD Class using CRUDBase ---
class CRUDDataset(CRUDBase[Dataset, DatasetCreateSchema, DatasetUpdateSchema]):
    def get_all_datasets_ordered_by_creation(
        self,
        db: Session,
        descending: bool = True
    ) -> List[Dataset]:
        query = db.query(Dataset)
        if descending:
            query = query.order_by(Dataset.created_at.desc(), Dataset.id.desc())
        else:
            query = query.order_by(Dataset.created_at.asc(), Dataset.id.asc())
        return query.all()
    
    async def get_dataset_analysis_dask(self, db: Session, dataset_id: int) -> Optional[DatasetAnalysisReport]:
        dataset_model = await asyncio.to_thread(db.query(Dataset).filter(Dataset.id == dataset_id).first)
        if not dataset_model:
            return None
        if not dataset_model.file_path:
            raise ValueError("Dataset has no file path")

        file_to_analyze = get_file_path(f'dataset_{dataset_id}_cleaned.csv')

        try:
            ddf = await asyncio.to_thread(load_csv_as_dask_dataframe, file_to_analyze)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file for analysis not found at: {file_to_analyze}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading dataset for analysis: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading dataset: {e}")

        if ddf.npartitions == 0 or len(ddf.columns) == 0:
            return DatasetAnalysisReport(
                total_records=0, total_features=len(ddf.columns), overall_missing_percentage=0,
                data_quality_score=0, features=[]
            )

        total_records_delayed = ddf.shape[0]
        missing_counts_delayed = ddf.isnull().sum()
        unique_counts_delayed = {col: ddf[col].nunique_approx() for col in ddf.columns}

        computed_values = await asyncio.to_thread(
            dask.compute,
            total_records_delayed,
            missing_counts_delayed,
            unique_counts_delayed
        )

        total_records = computed_values[0]
        missing_counts_series = computed_values[1]
        unique_counts_map = computed_values[2]

        total_features = len(ddf.columns)

        if total_records == 0:
            return DatasetAnalysisReport(
                total_records=0, total_features=total_features, overall_missing_percentage=0,
                data_quality_score=0,
                features=[
                    FeatureProfile(feature_name=str(col), dtype=str(ddf[col].dtype), missing_percentage=0, unique_percentage=0)
                    for col in ddf.columns
                ]
            )

        total_missing = missing_counts_series.sum()
        overall_missing_percentage = (total_missing / (total_records * total_features)) * 100 if total_features > 0 else 0


        features_analysis: List[FeatureProfile] = []
        # Since we have a check for total_records == 0 above and return,
        # we can assume total_records > 0 in this loop.
        for col in ddf.columns:
            missing_val_for_col = missing_counts_series[col]
            missing_pct = (missing_val_for_col / total_records) * 100

            unique_count_for_col = unique_counts_map[col]
            
            # Ensure approximate unique count doesn't exceed total records for percentage calculation.
            # nunique_approx() can sometimes yield a count slightly greater than total_records.
            # total_records is guaranteed to be > 0 here due to the earlier check.
            effective_unique_count = min(unique_count_for_col, total_records)
            
            unique_pct = (effective_unique_count / total_records) * 100

            features_analysis.append(
                FeatureProfile(
                    feature_name=str(col),
                    dtype=str(ddf[col].dtype),
                    missing_percentage=round(missing_pct, 2),
                    unique_percentage=round(unique_pct, 2) # This will now be capped at 100.0
                )
            )

        data_quality_score = round(100 - overall_missing_percentage, 2)
        actual_file_path_used_for_report = dataset_model.file_path
        project_name_to_report = dataset_model.project_name if dataset_model.project_name is not None else "N/A"
        
        return DatasetAnalysisReport(
            dataset_id=dataset_model.id,
            project_name=project_name_to_report,
            file_path=actual_file_path_used_for_report,
            total_records=total_records,
            total_features=total_features,
            overall_missing_percentage=round(overall_missing_percentage, 2),
            data_quality_score=data_quality_score,
            features=features_analysis
        )

# --- Instantiate the CRUD class ---
crud_dataset = CRUDDataset(Dataset)