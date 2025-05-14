"""
Automated Data Cleaning Tool

This script provides a function to automate common data cleaning tasks
on a CSV file based on user-selected options. It leverages scikit-learn
pipelines and custom transformers for a modular and robust approach.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from imblearn.over_sampling import SMOTE # Needs: pip install imbalanced-learn
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_categorical_dtype, is_string_dtype, is_datetime64_any_dtype
import sklearn

# --- Configuration & Setup ---

# Let's make sure we have the right scikit-learn version (1.2+) for cool features
if sklearn.__version__ < '1.2':
    raise ImportError(f"Scikit-learn version {sklearn.__version__} detected. "
                      f"This script needs scikit-learn 1.2 or later for the set_output API. "
                      f"Please upgrade: pip install -U scikit-learn")

# Setup friendly logging to see what's happening under the hood
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Helper Function ---

def _to_dataframe(X, feature_names_in):
    """
    A little helper to safely convert input data (NumPy array or DataFrame)
    into a pandas DataFrame inside our custom cleaning steps.
    It tries hard to keep the original row index and column names.
    """
    if isinstance(X, pd.DataFrame):
        # Already a DataFrame? Great, just pass it through.
        return X
    elif isinstance(X, np.ndarray):
        # It's a NumPy array. Let's try to make it a DataFrame.
        if feature_names_in is None:
             # We need column names! Warn the user if we don't have them.
             logging.warning("_to_dataframe: Input is NumPy array, but no column names provided. "
                             "Output DataFrame will have generic integer column names.")
             # Still try to keep the original index if possible
             return pd.DataFrame(X, index=getattr(X, 'index', None))
        else:
            # We have column names! Create the DataFrame.
            # Use .copy() to avoid unintended changes to the original array data later.
            return pd.DataFrame(X, columns=feature_names_in, index=getattr(X, 'index', None)).copy()
    else:
        # Don't know how to handle this type of input.
        raise TypeError(f"_to_dataframe: Expected pandas DataFrame or NumPy array, but got {type(X)}.")


# --- Custom Cleaning Steps (Transformers) ---
# We create our own cleaning tools that work nicely with scikit-learn pipelines.

class OutlierCapper(TransformerMixin, BaseEstimator):
    """
    Finds 'extreme' values (outliers) in numeric columns based on their
    Z-score (how many standard deviations away from the average they are)
    and 'caps' them, meaning it replaces them with a maximum allowed value.
    This is often gentler than removing outlier rows entirely.
    """
    def __init__(self, z_thresh=3.0):
        # How many standard deviations away counts as 'extreme'? Default is 3.
        self.z_thresh = z_thresh
        self.feature_names_in_ = None # We'll store column names we learned from
        self.means_ = None            # Store the average of each column
        self.stds_ = None             # Store the standard deviation of each column

    def fit(self, X, y=None):
        """Learn the mean and standard deviation for each numeric column."""
        # Make sure we're working with a DataFrame
        X_df = _to_dataframe(X, getattr(self, 'feature_names_in_', None))
        # Remember the column names we're seeing
        self.feature_names_in_ = X_df.columns.tolist()

        # Double-check if we accidentally got non-numeric data
        if not all(is_numeric_dtype(X_df[col]) for col in X_df.columns):
             logging.warning(f"{type(self).__name__}: Got some non-numeric columns during 'fit'. Will only process numeric ones.")

        # Focus only on the numeric columns provided
        numeric_cols_in_slice = [col for col in self.feature_names_in_ if is_numeric_dtype(X_df[col])]
        if not numeric_cols_in_slice:
             logging.warning(f"{type(self).__name__}: No numeric columns found to learn from.")
             self.means_ = np.array([])
             self.stds_ = np.array([])
             return self # Nothing more to learn

        try:
            # Calculate mean and standard deviation, ignoring missing values (NaNs)
            X_values = X_df[numeric_cols_in_slice].values
            self.means_ = np.nanmean(X_values, axis=0)
            self.stds_ = np.nanstd(X_values, axis=0)
            # Handle cases where a column has zero variation (std=0), which causes division errors.
            # We replace std=0 with NaN so Z-score calculation doesn't break.
            self.stds_[self.stds_ == 0] = np.nan
        except Exception as e:
            raise RuntimeError(f"Error calculating stats in {type(self).__name__} fit: {e}")

        return self # Ready to transform!

    def transform(self, X):
        """Apply the capping to the data."""
        # Make sure we're working with a DataFrame using the learned column names
        X_df = _to_dataframe(X, getattr(self, 'feature_names_in_', None))

        # Ensure the columns are in the same order as when we learned (fitted)
        if not X_df.columns.equals(pd.Index(self.feature_names_in_)):
             logging.warning(f"{type(self).__name__}: Columns changed order between fit and transform. Reordering.")
             X_df = X_df.reindex(columns=self.feature_names_in_)

        # Only process the numeric columns we learned about
        numeric_cols_in_slice = [col for col in self.feature_names_in_ if is_numeric_dtype(X_df[col])]
        # Check if we have anything to cap or if fitting failed
        if not numeric_cols_in_slice or self.means_ is None or self.stds_ is None or len(numeric_cols_in_slice) != len(self.means_):
             logging.warning(f"{type(self).__name__}: Nothing to transform (no numeric columns or fit failed). Returning original data.")
             return X_df.copy() # Return a copy to be safe

        # Work on a copy so we don't change the original DataFrame outside this function
        X_transformed = X_df.copy()

        # Get the numeric data as a NumPy array for faster calculations
        X_values = X_transformed[numeric_cols_in_slice].values
        means_values = np.asarray(self.means_)
        stds_values = np.asarray(self.stds_)

        # Calculate the upper and lower capping limits
        # Use nan_to_num for std dev in case some were NaN (zero variance columns)
        lower_bound = means_values - self.z_thresh * np.nan_to_num(stds_values, nan=0)
        upper_bound = means_values + self.z_thresh * np.nan_to_num(stds_values, nan=0)

        # Apply the capping using np.clip (efficiently handles boundaries and NaNs)
        capped_values = np.clip(X_values, lower_bound, upper_bound)

        # Put the capped values back into our copied DataFrame
        X_transformed[numeric_cols_in_slice] = capped_values

        return X_transformed # Return the DataFrame with outliers capped

    def get_feature_names_out(self, input_features=None):
        """Tells scikit-learn what the output column names are (they don't change)."""
        # Check if 'fit' was called and we stored the names
        if not hasattr(self, 'feature_names_in_') or self.feature_names_in_ is None:
            # If not fitted, try using input_features if provided as a fallback
            if input_features is not None:
                logging.warning(f"{type(self).__name__}.get_feature_names_out: Not fitted? Using input feature names.")
                return list(input_features) # Make sure it's a list
            else:
                raise ValueError(f"{type(self).__name__}: Cannot get output names before fitting.")
        # Otherwise, return the names we stored during fit
        return self.feature_names_in_


class NoiseSmoother(TransformerMixin, BaseEstimator):
    """
    Smooths out 'noisy' fluctuations in numeric data using a rolling average.
    Helpful for time-series or sequential data where temporary spikes might
    not be meaningful. Assumes the row order (index) is significant.
    """
    def __init__(self, window=3):
        # How many data points to average over? Default is 3 (current point and previous 2).
        self.window = window
        self.feature_names_in_ = None # Store column names learned during fit

    def fit(self, X, y=None):
        """Learns which columns to apply smoothing to."""
        X_df = _to_dataframe(X, getattr(self, 'feature_names_in_', None))
        self.feature_names_in_ = X_df.columns.tolist()
        # Check if we got non-numeric data
        if not all(is_numeric_dtype(X_df[col]) for col in X_df.columns):
            logging.warning(f"{type(self).__name__}: Received non-numeric columns. Smoothing only applies to numeric ones.")
        return self # Nothing else to learn

    def transform(self, X):
        """Applies the rolling mean smoothing."""
        X_df = _to_dataframe(X, getattr(self, 'feature_names_in_', None))
        # Ensure column order matches fit
        if not X_df.columns.equals(pd.Index(self.feature_names_in_)):
             logging.warning(f"{type(self).__name__}: Column order mismatch. Reordering.")
             X_df = X_df.reindex(columns=self.feature_names_in_)

        # Focus on numeric columns
        numeric_cols_in_slice = [col for col in self.feature_names_in_ if is_numeric_dtype(X_df[col])]
        if not numeric_cols_in_slice:
             logging.warning(f"{type(self).__name__}: No numeric columns to smooth. Returning original.")
             return X_df.copy()

        X_transformed = X_df.copy()
        original_index = X_transformed.index # Keep track of the original row order

        # Apply rolling mean directly using pandas (handles indices nicely)
        # Sort by index first to ensure rolling window is consistent, even if data wasn't sorted initially.
        X_numeric_subset = X_transformed[numeric_cols_in_slice].sort_index()
        smoothed_values = X_numeric_subset.rolling(window=self.window, min_periods=1).mean()

        # Rolling mean creates NaNs at the beginning. Fill them reasonably.
        # bfill fills with next valid value, ffill fills with previous valid value.
        smoothed_values = smoothed_values.fillna(method='bfill').fillna(method='ffill')

        # Put smoothed values back, ensuring they align with the *original* index order
        X_transformed.loc[:, numeric_cols_in_slice] = smoothed_values.reindex(original_index)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Output column names are the same as input."""
        if not hasattr(self, 'feature_names_in_') or self.feature_names_in_ is None:
            if input_features is not None:
                logging.warning(f"{type(self).__name__}.get_feature_names_out: Not fitted? Using input feature names.")
                return list(input_features)
            else:
                raise ValueError(f"{type(self).__name__}: Cannot get output names before fitting.")
        return self.feature_names_in_


class CardinalityReducer(TransformerMixin, BaseEstimator):
    """
    Reduces the number of unique categories in text/categorical columns.
    It keeps the most frequent categories and groups the rest into an 'Other' category.
    This helps prevent issues with models when some categories appear very rarely.
    """
    def __init__(self, max_unique=20):
        # How many top categories to keep? Default is 20.
        self.max_unique = max_unique
        self.top_categories_ = {}    # Stores the top categories for each column learned during fit
        self.feature_names_in_ = None # Stores column names learned during fit

    def fit(self, X, y=None):
        """Learns the most frequent categories for each relevant column."""
        X_df = _to_dataframe(X, getattr(self, 'feature_names_in_', None))
        self.feature_names_in_ = X_df.columns.tolist()

        # Need string column names to proceed reliably
        if X_df.columns is None or not all(isinstance(c, str) for c in X_df.columns):
             raise RuntimeError(f"{type(self).__name__}: Input must have string column names for fit.")

        # Check dtypes (warning only)
        relevant_col_types = (is_object_dtype, is_categorical_dtype, lambda c: c.dtype == 'bool', is_string_dtype)
        if not any(check(X_df[col]) for col in X_df.columns for check in relevant_col_types if col in X_df.columns): # check if col exists
            logging.warning(f"{type(self).__name__}: Received potentially non-categorical/object/bool/string data during fit.")

        self.top_categories_ = {} # Reset if fit is called again
        # Identify columns suitable for cardinality reduction
        target_cols_in_slice = [
            col for col in self.feature_names_in_
            if col in X_df.columns and any(check(X_df[col]) for check in relevant_col_types) # ensure col exists before checking type
        ]


        if not target_cols_in_slice:
             logging.warning(f"{type(self).__name__}: No suitable categorical/object/bool/string columns found to fit.")
             return self

        # Find top categories for each suitable column
        for col in target_cols_in_slice:
            # No need to check 'if col in X_df.columns' again, already done above
            try:
                # Count occurrences of each unique value (including NaN)
                counts = X_df[col].value_counts(dropna=False)
                # Get the names of the top N categories
                if self.max_unique > 0:
                     top = counts.nlargest(self.max_unique).index.tolist()
                else: # If max_unique is 0 or less, group everything into 'Other'
                     top = []
                     logging.warning(f"max_unique <= 0 for column '{col}'. All non-NaN values will become 'Other'.")
                # Store as a set for faster checking later
                self.top_categories_[col] = set(top)
            except Exception as e:
                 logging.error(f"Error finding top categories for column '{col}': {e}. Skipping reduction for this column.")
                 self.top_categories_[col] = set() # Mark as failed/empty
        return self

    def transform(self, X):
        """Applies the grouping into 'Other'."""
        X_df = _to_dataframe(X, getattr(self, 'feature_names_in_', None))
        # Ensure column order matches fit
        if not X_df.columns.equals(pd.Index(self.feature_names_in_)):
             logging.warning(f"{type(self).__name__}: Column order mismatch. Reordering.")
             X_df = X_df.reindex(columns=self.feature_names_in_)

        # Need string column names again
        if X_df.columns is None or not all(isinstance(c, str) for c in X_df.columns):
             logging.error(f"{type(self).__name__}: Input must have string column names for transform. Returning original.")
             return X_df.copy()

        # Check if fit actually found any categories to reduce
        if not self.top_categories_:
             logging.warning(f"{type(self).__name__}: No categories learned during fit. Skipping transform.")
             return X_df.copy()

        Xc = X_df.copy() # Work on a copy

        # Apply reduction only to columns we learned about and that exist in the current data
        mappable_cols = [col for col in self.top_categories_.keys() if col in Xc.columns]
        if not mappable_cols:
            logging.warning(f"{type(self).__name__}: Fitted columns not found in transform input. Skipping.")
            return Xc

        for col in mappable_cols:
             top_cats = self.top_categories_[col]
             # Check if the value in the column is NOT one of the top categories AND is not NaN
             # .isin() is efficient for checking against the set of top categories
             is_rare_mask = ~Xc[col].isin(top_cats) & Xc[col].notna()

             if is_rare_mask.any(): # Only modify if there are rare values to replace
                 # Replace these 'rare' (not top) values with 'Other'
                 # Ensure consistency, convert column to object type if mixing strings and others
                 if not is_object_dtype(Xc[col]):
                    try:
                        Xc[col] = Xc[col].astype(object)
                    except Exception as e:
                        logging.error(f"Could not convert column '{col}' to object type for 'Other' replacement: {e}. Skipping reduction for this column.")
                        continue # Skip to next column if conversion fails
                 Xc.loc[is_rare_mask, col] = 'Other'


        return Xc

    def get_feature_names_out(self, input_features=None):
        """Output column names are the same as input."""
        if not hasattr(self, 'feature_names_in_') or self.feature_names_in_ is None:
            if input_features is not None:
                logging.warning(f"{type(self).__name__}.get_feature_names_out: Not fitted? Using input feature names.")
                return list(input_features)
            else:
                raise ValueError(f"{type(self).__name__}: Cannot get output names before fitting.")
        return self.feature_names_in_


# --- Pipeline Builder ---
# This function assembles the cleaning steps into a scikit-learn pipeline
# based on which tasks the user wants to perform.

def build_pipeline(numeric_cols, categorical_cols, flags, internal_params):
    """Builds the scikit-learn preprocessing pipeline."""
    logging.info("Building the cleaning pipeline...")

    # Get specific settings for steps from internal_params
    numeric_missing_strategy = internal_params.get('numeric_missing_strategy', 'mean')
    smooth_window = internal_params.get('smooth_window', 3)
    outlier_z = internal_params.get('outlier_z', 3.0)
    max_unique_categories = internal_params.get('max_unique_categories', 20)

    # --- Define steps for NUMERIC columns ---
    num_steps = []
    if flags.get('handle_missing_values'):
        logging.debug(f"Adding: Numeric Imputation (strategy={numeric_missing_strategy})")
        num_steps.append(('impute', SimpleImputer(strategy=numeric_missing_strategy)))
    if flags.get('smooth_noisy_data'):
        logging.debug(f"Adding: Noise Smoothing (window={smooth_window})")
        num_steps.append(('smooth', NoiseSmoother(window=smooth_window)))
    if flags.get('handle_outliers'):
        logging.debug(f"Adding: Outlier Capping (z_thresh={outlier_z})")
        num_steps.append(('cap', OutlierCapper(z_thresh=outlier_z)))
    if flags.get('feature_scaling'):
        logging.debug("Adding: Feature Scaling (StandardScaler)")
        num_steps.append(('scale', StandardScaler()))

    # --- Define steps for CATEGORICAL columns ---
    cat_steps = []
    if flags.get('handle_missing_values'):
        logging.debug("Adding: Categorical Imputation (strategy='most_frequent')")
        cat_steps.append(('impute_cat', SimpleImputer(strategy='most_frequent')))
    if flags.get('reduce_cardinality'):
        logging.debug(f"Adding: Cardinality Reduction (max_unique={max_unique_categories})")
        cat_steps.append(('cardinal', CardinalityReducer(max_unique=max_unique_categories)))
    if flags.get('encode_categorical_values'):
        logging.debug("Adding: One-Hot Encoding")
        # OHE converts categories to 0s and 1s. sparse=False gives a standard NumPy array/DataFrame.
        cat_steps.append(('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

    # --- Combine pipelines using ColumnTransformer ---
    transformers_config = []

    # Create numeric pipeline if there are numeric columns and steps requested
    if numeric_cols and num_steps:
        num_pipeline = Pipeline(num_steps)
        num_pipeline.set_output(transform='pandas') # Ensure output is a DataFrame
        # *** CORRECTED: Append only (pipeline, columns) tuple for make_column_transformer ***
        transformers_config.append((num_pipeline, numeric_cols))
        logging.info(f"Numeric pipeline created for columns: {numeric_cols}")

    # Create categorical pipeline if there are categorical columns and steps requested
    if categorical_cols and cat_steps:
        cat_pipeline = Pipeline(cat_steps)
        cat_pipeline.set_output(transform='pandas') # Ensure output is a DataFrame
        # *** CORRECTED: Append only (pipeline, columns) tuple for make_column_transformer ***
        transformers_config.append((cat_pipeline, categorical_cols))
        logging.info(f"Categorical pipeline created for columns: {categorical_cols}")

    # If no steps were defined for any column type, return None
    if not transformers_config:
        logging.warning("No cleaning steps were enabled or no relevant columns found. "
                        "No pipeline will be built.")
        return None

    # make_column_transformer applies specific pipelines to specific columns
    # It expects tuples of (transformer, columns) when using *args unpacking.
    ct = make_column_transformer(
        *transformers_config, # Unpack the list of (pipeline, columns) tuples
        remainder='drop',
        verbose_feature_names_out=False # Keep output names clean
    )
    # Ensure the final output of the ColumnTransformer is also a DataFrame
    ct.set_output(transform='pandas')
    logging.info("ColumnTransformer assembled.")
    return ct


# --- Main Automation Function ---

def automate_cleaning_by_task(
    input_csv: str,
    output_csv: str,
    target_col: str = None,
    remove_duplicates: bool = False,
    handle_missing_values: bool = False,
    smooth_noisy_data: bool = False,
    handle_outliers: bool = False,
    reduce_cardinality: bool = False,
    encode_categorical_values: bool = False,
    feature_scaling: bool = False,
    handle_imbalanced_data: bool = False,
    **read_csv_kwargs # Pass extra options to pandas read_csv if needed
):
    """
    Orchestrates the automated data cleaning process based on selected tasks.

    It loads data, optionally removes duplicates, identifies column types,
    builds and applies a cleaning pipeline, optionally balances the data using SMOTE,
    and saves the cleaned data.

    Args:
        input_csv: Path to the input CSV file.
        output_csv: Path where the cleaned CSV file will be saved.
        target_col: Name of the target variable column (needed for SMOTE).
        remove_duplicates: If True, remove duplicate rows.
        handle_missing_values: If True, impute missing values.
        smooth_noisy_data: If True, apply rolling mean smoothing to numeric columns.
        handle_outliers: If True, cap numeric outliers using Z-score.
        reduce_cardinality: If True, group rare categories into 'Other'.
        encode_categorical_values: If True, apply One-Hot Encoding to categorical columns.
        feature_scaling: If True, apply StandardScaler to numeric columns.
        handle_imbalanced_data: If True, apply SMOTE to balance the target variable.
        **read_csv_kwargs: Additional arguments for pd.read_csv (e.g., sep=';').
    """
    logging.info(f"--- Starting Automated Cleaning Process ---")
    logging.info(f"Input file: {input_csv}")
    logging.info(f"Output file: {output_csv}")

    # Internal settings for the cleaning steps (could be customized further)
    internal_params = {
        'numeric_missing_strategy': 'mean', # How to fill missing numbers
        'smooth_window': 3,                 # Window size for smoothing
        'outlier_z': 3.0,                   # Z-score threshold for outliers
        'max_unique_categories': 20,        # Max categories to keep before grouping
        'high_cardinality_warning_threshold': 100, # Warn if >100 unique categories
    }

    # == Step 1: Load Data ==
    logging.info("Step 1: Loading data...")
    try:
        # low_memory=False can help prevent dtype guessing issues with large files
        df = pd.read_csv(input_csv, low_memory=False, **read_csv_kwargs)
        logging.info(f"Data loaded successfully. Initial shape: {df.shape}")
        # Log initial data types for reference
        # logging.debug(f"Initial dtypes:\n{df.dtypes}") # Optional: more detail
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {input_csv}: {e}")
        raise

    original_cols = df.columns.tolist() # Keep track of original columns

    # == Step 2: Remove Duplicates (Optional) ==
    logging.info("Step 2: Checking for duplicates...")
    if remove_duplicates:
        initial_rows = len(df)
        logging.info(f"-> Option enabled. Removing duplicate rows (keeping first occurrence).")
        df.drop_duplicates(inplace=True, ignore_index=True) # Use ignore_index to reset index after drops
        rows_after_drop = len(df)
        if rows_after_drop < initial_rows:
            logging.info(f"   Removed {initial_rows - rows_after_drop} duplicate rows. Shape is now: {df.shape}")
        else:
            logging.info("   No duplicate rows found.")
    else:
        logging.info("-> Option disabled. Skipping duplicate removal.")

    # == Step 3: Separate Target Variable (if SMOTE is requested) ==
    logging.info("Step 3: Handling target variable...")
    y = None # Initialize target variable as None
    features_df = df # Start with the full DataFrame (or after duplicate removal)

    if handle_imbalanced_data:
        logging.info("-> Balancing requested. Separating target column.")
        if not target_col or target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' must be provided and exist "
                             f"in the data when 'handle_imbalanced_data' is True.")

        y = features_df[target_col].copy() # Get the target column
        features_df = features_df.drop(columns=[target_col]) # Remove target from features
        logging.info(f"   Target column '{target_col}' separated. Features shape: {features_df.shape}")
        logging.info(f"   Target value counts:\n{y.value_counts().to_string()}") # Show class distribution

        # --- Sanity checks for SMOTE ---
        num_unique_targets = y.nunique()
        if num_unique_targets > 2:
             # Standard SMOTE is designed for binary tasks. Warn if multi-class.
             logging.warning(f"   Target '{target_col}' has {num_unique_targets} unique values. "
                             f"Standard SMOTE works best for binary classification. Results might be unexpected.")
        elif num_unique_targets < 2:
             # Can't balance if there's only one class!
             logging.warning(f"   Target '{target_col}' has only {num_unique_targets} unique value. "
                             f"SMOTE cannot be applied. Disabling balancing.")
             handle_imbalanced_data = False # Turn off the flag
        elif num_unique_targets == 2:
             # Check if the minority class has enough samples for SMOTE's default setting (k=5)
             min_samples = y.value_counts().min()
             k_neighbors_smote = 5
             if min_samples <= k_neighbors_smote:
                 logging.warning(f"   Minority class in '{target_col}' has only {min_samples} samples. "
                                 f"SMOTE requires more samples than neighbors (default k={k_neighbors_smote}). Disabling balancing.")
                 handle_imbalanced_data = False # Turn off the flag
    else:
        logging.info("-> Balancing not requested or target not specified.")
        # If target_col was provided but balancing wasn't requested, we might still want
        # to ensure it's not processed by the feature pipeline. Let's check.
        if target_col and target_col in features_df.columns:
            logging.info(f"   Target column '{target_col}' exists but balancing is off. It will be kept but ignored by the feature pipeline if not numeric/categorical.")


    # == Step 4: Detect Feature Types ==
    logging.info("Step 4: Detecting numeric and categorical features...")
    # Numeric columns (but not simple True/False ones)
    discovered_numeric_cols = [col for col in features_df.columns if is_numeric_dtype(features_df[col]) and features_df[col].dtype != 'bool']
    # Categorical columns (text, object, category, True/False)
    discovered_categorical_cols = [col for col in features_df.columns if is_object_dtype(features_df[col]) or is_categorical_dtype(features_df[col]) or features_df[col].dtype == 'bool' or is_string_dtype(features_df[col])]

    processed_cols_set = set(discovered_numeric_cols + discovered_categorical_cols)
    # Find columns that weren't classified as numeric or categorical (e.g., datetime)
    ignored_cols = [col for col in features_df.columns if col not in processed_cols_set]

    logging.info(f"   Detected {len(discovered_numeric_cols)} numeric features: {discovered_numeric_cols}")
    logging.info(f"   Detected {len(discovered_categorical_cols)} categorical/boolean features: {discovered_categorical_cols}")
    if ignored_cols:
        # Specifically mention datetimes if any are ignored
        datetime_cols = [col for col in ignored_cols if is_datetime64_any_dtype(features_df[col])]
        other_ignored = [col for col in ignored_cols if col not in datetime_cols]
        if datetime_cols: logging.info(f"   Ignoring {len(datetime_cols)} datetime features: {datetime_cols}")
        if other_ignored: logging.info(f"   Ignoring {len(other_ignored)} other feature types: {other_ignored}")


    # == Step 5: Drop Constant/Zero-Variance Features ==
    # These columns provide no information for models.
    logging.info("Step 5: Removing constant or zero-variance features...")
    final_numeric_cols = discovered_numeric_cols[:] # Work with copies
    final_categorical_cols = discovered_categorical_cols[:]

    # Check numeric columns for zero variance (or all NaN)
    drop_num_vz = []
    if final_numeric_cols:
        num_subset = features_df[[col for col in final_numeric_cols if col in features_df.columns]] # Ensure columns still exist
        if not num_subset.empty and len(num_subset) > 1:
             variances = num_subset.var()
             # Columns where variance is 0 (or NaN treated as 0) are constant
             drop_num_vz = variances[variances.fillna(0) == 0].index.tolist()

    # Check categorical columns for only one unique value (including NaN)
    drop_cat_const = []
    if final_categorical_cols:
        cat_subset = features_df[[col for col in final_categorical_cols if col in features_df.columns]] # Ensure columns still exist
        if not cat_subset.empty:
             for col in cat_subset.columns: # Iterate over existing columns
                  if features_df[col].nunique(dropna=False) <= 1:
                      drop_cat_const.append(col)

    # Combine columns to drop
    cols_to_drop_early = drop_num_vz + drop_cat_const
    if cols_to_drop_early:
        # Ensure we only try to drop columns that actually still exist
        cols_to_drop_early = [col for col in cols_to_drop_early if col in features_df.columns]
        if cols_to_drop_early: # Check if list is still non-empty
            logging.info(f"   Found columns with no variation: {cols_to_drop_early}. Dropping them.")
            features_df.drop(columns=cols_to_drop_early, inplace=True)
            # Update our lists of columns to reflect the drops
            final_numeric_cols = [c for c in final_numeric_cols if c not in cols_to_drop_early]
            final_categorical_cols = [c for c in final_categorical_cols if c not in cols_to_drop_early]
            logging.info(f"   Features shape after dropping constant columns: {features_df.shape}")
        else:
            logging.info("   No constant or zero-variance columns found among existing features.")
    else:
         logging.info("   No constant or zero-variance columns found.")

    logging.info(f"   Final numeric columns for pipeline: {final_numeric_cols}")
    logging.info(f"   Final categorical columns for pipeline: {final_categorical_cols}")

    # == Step 6: High Cardinality Check (Warning) ==
    # Warn if categorical columns have too many unique values, as OHE can explode the dataset size.
    logging.info("Step 6: Checking for high cardinality in categorical features...")
    high_card_threshold = internal_params['high_cardinality_warning_threshold']
    max_unique_reduc = internal_params['max_unique_categories']
    if final_categorical_cols and (reduce_cardinality or encode_categorical_values):
        high_card_cols = []
        # Check columns that still exist after constant dropping
        valid_cat_cols_check = [col for col in final_categorical_cols if col in features_df.columns]
        if valid_cat_cols_check:
            try:
                high_card_cols = [
                    col for col in valid_cat_cols_check
                    if features_df[col].nunique() > high_card_threshold
                ]
            except Exception as e:
                logging.warning(f"   Could not perform high cardinality check due to error: {e}")

        if high_card_cols:
            action_msg = ""
            if encode_categorical_values: action_msg += " OneHotEncoding"
            if reduce_cardinality: action_msg += f" (reducing to {max_unique_reduc})"
            logging.warning(f"   High Cardinality Alert (> {high_card_threshold} unique values) in: {high_card_cols}."
                            f"{action_msg} may significantly increase dataset width.")
        else:
            logging.info("   No high cardinality features detected.")
    else:
        logging.info("   Skipping high cardinality check (no relevant columns or tasks).")


    # == Step 7: Build and Apply Cleaning Pipeline ==
    logging.info("Step 7: Building and applying the main cleaning pipeline...")
    # Collect the boolean flags for the pipeline builder
    flags_for_pipeline = {
        'handle_missing_values': handle_missing_values,
        'smooth_noisy_data': smooth_noisy_data,
        'handle_outliers': handle_outliers,
        'reduce_cardinality': reduce_cardinality,
        'encode_categorical_values': encode_categorical_values,
        'feature_scaling': feature_scaling
    }

    # Build the pipeline using our function
    preprocessor = build_pipeline(final_numeric_cols, final_categorical_cols, flags_for_pipeline, internal_params)

    # Apply the pipeline (if one was built)
    processed_features_df = features_df.copy() # Start with the current features
    if preprocessor is None:
         logging.warning("   No pipeline built (no steps enabled or no relevant columns). "
                         "Using data after previous steps (duplicate/constant removal).")
    else:
        logging.info(f"   Applying pipeline to {len(features_df)} samples and {len(features_df.columns)} features.")
        try:
            # This is where the main transformations happen!
            processed_features_df = preprocessor.fit_transform(features_df)
            logging.info(f"   Pipeline transformation complete. Processed features shape: {processed_features_df.shape}")
            # Log column names after transformation (useful after OHE)
            logging.debug(f"   Columns after pipeline: {processed_features_df.columns.tolist()}")
        except Exception as e:
            logging.error(f"   Error during pipeline processing: {e}")
            raise # Stop execution if the pipeline fails

    # == Step 8: Balance Data using SMOTE (Optional) ==
    logging.info("Step 8: Handling data balancing (SMOTE)...")
    final_df = None # Initialize the final DataFrame

    # Check if balancing is still enabled (it might have been disabled by sanity checks)
    if handle_imbalanced_data and y is not None:
        logging.info(f"   Attempting SMOTE balancing for target '{target_col}'.")
        # --- Final check before SMOTE ---
        # SMOTE needs numeric features. Let's verify.
        non_numeric_for_smote = [
            col for col in processed_features_df.columns
            if not is_numeric_dtype(processed_features_df[col])
        ]
        if non_numeric_for_smote:
            # If we have non-numeric columns here, SMOTE will likely fail.
            logging.error(f"   SMOTE requires all numeric features, but found non-numeric columns "
                          f"after pipeline: {non_numeric_for_smote}. "
                          f"Ensure 'encode_categorical_values=True' was used for categorical data. "
                          f"Cannot apply SMOTE.")
            # Fallback: Don't balance, just use the processed features.
            handle_imbalanced_data = False # Mark balancing as skipped
            final_df = processed_features_df # Use the features as they are
        else:
            # --- Apply SMOTE ---
            # Ensure features and target have the same number of samples before SMOTE
            if len(processed_features_df) != len(y):
                 # Adding check for index alignment as well before raising error
                 if not processed_features_df.index.equals(y.index):
                      logging.warning("   Index mismatch between features and target before SMOTE. Attempting to align by resetting index.")
                      processed_features_df = processed_features_df.reset_index(drop=True)
                      y = y.reset_index(drop=True)
                      if len(processed_features_df) != len(y): # Check length again after reset
                          raise RuntimeError(f"Mismatch in feature ({len(processed_features_df)}) and target ({len(y)}) "
                                             f"lengths even after index reset before SMOTE.")
                 else: # Indices match but lengths don't - indicates prior error
                     raise RuntimeError(f"Mismatch in feature ({len(processed_features_df)}) and target ({len(y)}) "
                                     f"lengths before SMOTE despite matching indices. This shouldn't happen.")

            try:
                logging.info("   Applying SMOTE...")
                smote = SMOTE(random_state=42) # Use random_state for reproducible results
                # imblearn >= 0.10 returns DataFrames/Series if input is such
                X_resampled_df, y_resampled_series = smote.fit_resample(processed_features_df, y)
                logging.info(f"   SMOTE complete. Resampled features shape: {X_resampled_df.shape}")
                logging.info(f"   Resampled target value counts:\n{y_resampled_series.value_counts().to_string()}")

                # Combine the balanced features and target
                y_resampled_series = y_resampled_series.rename(target_col) # Ensure target Series has the right name
                final_df = pd.concat([X_resampled_df, y_resampled_series], axis=1)
                logging.info(f"   Balanced data combined. Final shape: {final_df.shape}")

            except ValueError as ve:
                # Catch common SMOTE errors (e.g., k_neighbors > samples)
                logging.error(f"   SMOTE failed with ValueError: {ve}. This often relates to "
                              f"insufficient samples in the minority class. Skipping balancing.")
                handle_imbalanced_data = False # Mark balancing as skipped
                final_df = processed_features_df # Fallback to processed features
            except Exception as e:
                 logging.error(f"   Unexpected error during SMOTE: {e}")
                 raise # Re-raise other unexpected errors

    # --- Combine Features and Target if SMOTE was NOT run ---
    if not handle_imbalanced_data:
        if final_df is None: # If SMOTE didn't run or failed, final_df wasn't assigned
             final_df = processed_features_df # Use the features processed by the pipeline

        # Now, add the original target 'y' back if it exists
        if y is not None:
            logging.info("   SMOTE not applied or failed. Re-adding original target column.")
            # Check if target is already present (could happen in some error flows)
            if target_col not in final_df.columns:
                try:
                    # Ensure y Series has the correct name
                    y_to_concat = y.rename(target_col)
                    # Concatenate based on index alignment.
                    # Both final_df (from pipeline with set_output) and y (original split)
                    # should share the same index from before target separation (unless duplicates were dropped).
                    # If duplicates were dropped, indices might not align perfectly. Resetting might be safer.
                    # Let's check index equality first.
                    if not final_df.index.equals(y_to_concat.index):
                         logging.warning("   Indices between processed features and target do not align. "
                                         "This might happen if duplicates were removed. Resetting index before combining.")
                         final_df = pd.concat([final_df.reset_index(drop=True), y_to_concat.reset_index(drop=True)], axis=1)
                    else:
                         # Indices align, direct concat is safe
                         final_df = pd.concat([final_df, y_to_concat], axis=1)

                    logging.info(f"   Original target '{target_col}' re-added. Final shape: {final_df.shape}")
                except Exception as e:
                     # Log error if concatenation fails (e.g., index mismatch somehow)
                     logging.error(f"   Error re-adding original target column: {e}. Output will lack target.")
            else:
                logging.info(f"   Target column '{target_col}' seems already present. Shape: {final_df.shape}")
        else:
             logging.info("   No target column specified initially. Final data contains only processed features.")


    # == Step 9: Save Cleaned Data ==
    logging.info("Step 9: Saving cleaned data...")
    try:
        if final_df is None:
             logging.error("   Final DataFrame ('final_df') was not created due to errors. Cannot save.")
             raise NameError("final_df is not defined")

        final_df.to_csv(output_csv, index=False) # index=False is standard for cleaned data CSVs
        logging.info(f"   Data saved successfully to: {output_csv}")
    except Exception as e:
        logging.error(f"   Error saving data to {output_csv}: {e}")
        raise

    logging.info(f"--- Automated Cleaning Process Finished ---")