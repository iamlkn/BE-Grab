import pandas as pd
import pycaret.classification as pyclf
import pycaret.regression as pyreg
from sklearn.pipeline import Pipeline # Needed for SHAP preprocessor separation
import time
import numpy as np
import os
import joblib
import shutil # Needed for moving files
import mlflow
import datetime
import logging
import yaml # For loading config
import json # For saving task_type metadata and other JSONs
import re # For sanitize_filename
from typing import List, Optional, Dict, Any, Tuple, Union
import pycaret # Keep for version check if needed elsewhere
import inspect # Needed for _log_plot_artifact
import matplotlib.pyplot as plt # Needed for _log_plot_artifact workaround
from io import StringIO
import json


try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None
    logging.warning("ydata-profiling not found. Run 'pip install ydata-profiling' to enable data profiling.")

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    Report = None
    DataDriftPreset = None
    logging.warning("evidently not found. Run 'pip install evidently' to enable data drift detection.")

try:
    import shap
except ImportError:
    shap = None
    logging.warning("shap not found. Run 'pip install shap' to enable prediction explanations.")


# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - [%(funcName)s] - %(message)s'
)
logger = logging.getLogger(__name__) # Use module-specific logger

# --- Helper Functions ---

# Get the directory where this runner.py file lives
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_CURRENT_DIR, 'config.yaml')

def sanitize_filename(name: str) -> str:
    """Removes or replaces characters unsuitable for filenames."""
    # Remove most special characters, replace spaces/dots with underscore
    name = re.sub(r'[^\w\-. ]', '', name) # Keep word chars, hyphen, dot, space
    name = re.sub(r'[ .]+', '_', name) # Replace spaces/dots with underscore
    return name

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    Uses a default path relative to this script if none provided.
    Validates essential keys and sets defaults for new features.
    """
    if not os.path.exists(config_path) and config_path != DEFAULT_CONFIG_PATH:
         logger.warning(f"Provided config path '{config_path}' not found. Falling back to default: '{DEFAULT_CONFIG_PATH}'")
         config_path = DEFAULT_CONFIG_PATH
    elif not os.path.exists(config_path):
         logger.error(f"Default configuration file not found at '{config_path}'")
         raise FileNotFoundError(f"Default configuration file not found at '{config_path}'")

    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
             logger.error(f"Invalid or empty configuration file: {config_path}")
             raise ValueError(f"Configuration file {config_path} is not a valid YAML dictionary.")
        logger.info("Configuration loaded successfully.")

        if 'session_id' not in config or config.get('session_id') is None:
            config['session_id'] = int(time.time())
            logger.info(f"Using generated session_id: {config['session_id']}")

        if 'output_base_dir' not in config or not config.get('output_base_dir'):
            default_output_dir = os.path.join(os.getcwd(), 'automl_outputs_default')
            logger.warning(f"output_base_dir not specified. Defaulting to CWD-relative path: '{default_output_dir}'.")
            config['output_base_dir'] = default_output_dir

        config.setdefault('run_data_profiling', False)
        config.setdefault('profile_report_name', "data_profile_report.html")
        config.setdefault('enable_drift_check_on_predict', True)
        config.setdefault('drift_report_name', "prediction_drift_report.html")
        config.setdefault('generate_model_card', False)
        config.setdefault('register_model_in_mlflow', False)
        config.setdefault('mlflow_registered_model_name', "AutoML_Step_Based_Model")
        config.setdefault('mlflow_model_stage', "Staging")
        config.setdefault('enable_prediction_explanation', False)
        config.setdefault('shap_enabled_if_possible', False) # More specific control for SHAP in analysis step
        config.setdefault('optimize_pandas_dtypes', True) 
        config.setdefault('use_sampling_in_setup', False) 
        config.setdefault('save_baseline_models', False)
        config.setdefault('analyze_tuned_step2', True) 

        config.setdefault('setup_params_extra', {}) # To pass arbitrary params to setup
        config.setdefault('tuning_search_library', 'scikit-learn') # Default tuning lib
        config.setdefault('tuning_search_algorithm', 'random') # Default tuning algo
        config.setdefault('use_gpu', False)

        # --- Validate essential keys needed by the runner ---
        required_keys_for_runner = [
            "session_id", "output_base_dir", "experiment_name", "mlflow_tracking_uri",
            "data_file_path", "target_column",
            "numeric_imputation", "unique_value_threshold_for_classification",
            "sort_metric_classification", "optimize_metric_classification",
            "sort_metric_regression", "optimize_metric_regression",
            "baseline_folds", "tuning_folds", "tuning_iterations"
        ]
        missing_keys = [k for k in required_keys_for_runner if k not in config] # Check only required ones
        if missing_keys:
            logger.warning(f"Config might be missing keys expected by runner steps: {missing_keys}")

        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file '{config_path}': {e}", exc_info=True)
        raise ValueError(f"Invalid YAML syntax in {config_path}") from e
    except Exception as e:
        logger.error(f"Error loading or validating configuration from '{config_path}': {e}", exc_info=True)
        raise

def optimize_dtypes(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame: # Add config parameter
    """
    (Helper) Downcast numeric columns and convert low-cardinality objects to 'category'.
    Reduces memory usage. Handles NA values using Pandas nullable types.
    Requires config dictionary for category_conversion_threshold.
    """
    logger.info("Optimizing DataFrame dtypes for memory efficiency...")
    start_mem = df.memory_usage(deep=True).sum() / 1024**2 # Use deep=True for accurate object memory
    df_opt = df.copy() # Work on a copy

    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        numerics = pd.api.types.is_numeric_dtype(col_type)
        contains_na = df_opt[col].isnull().any()

        if numerics:
            # ... (numeric handling code remains the same) ...
             if df_opt[col].isnull().all():
                logger.debug(f"Column '{col}' is all NA. Skipping numeric optimization.")
                continue
             try:
                c_min = df_opt[col].min(skipna=True)
                c_max = df_opt[col].max(skipna=True)
             except TypeError:
                logger.warning(f"Could not compute min/max for numeric column '{col}'. Skipping optimization.")
                continue

             if pd.isna(c_min) or pd.isna(c_max):
                logger.debug(f"Column '{col}' min/max calculation resulted in NA. Skipping numeric downcasting.")
                continue

             is_integer_like_float = False
             if col_type.kind == 'f':
                 non_na_series = df_opt[col].dropna()
                 if not non_na_series.empty:
                      try: # Avoid errors if non-numeric values slipped into float column
                          is_integer_like_float = (np.mod(non_na_series, 1) == 0).all()
                      except TypeError:
                          is_integer_like_float = False # Cannot determine

             # Integer downcasting
             if col_type.kind == 'i' or col_type.kind == 'u' or is_integer_like_float:
                try:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max: target_type = pd.Int8Dtype() if contains_na else np.int8
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max: target_type = pd.Int16Dtype() if contains_na else np.int16
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max: target_type = pd.Int32Dtype() if contains_na else np.int32
                    else: target_type = pd.Int64Dtype() if contains_na else np.int64

                    current_dtype_str = str(df_opt[col].dtype)
                    target_dtype_str = str(target_type)

                    # Only convert if target type is different
                    if current_dtype_str != target_dtype_str:
                        try:
                            df_opt[col] = df_opt[col].astype(target_type)
                            logger.debug(f"Column '{col}' converted from {current_dtype_str} to {target_dtype_str}.")
                        except (TypeError, OverflowError) as e:
                            logger.warning(f"Could not convert column '{col}' to {target_dtype_str}. Keeping {current_dtype_str}. Error: {e}")
                except TypeError as te: # Error during np.iinfo comparison
                    logger.warning(f"TypeError during integer range comparison for column '{col}'. Keeping {df_opt[col].dtype}. Error: {te}")

             # Float downcasting (only to float32)
             elif col_type.kind == 'f' and df_opt[col].dtype != np.float32:
                 try:
                     if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                         df_opt[col] = df_opt[col].astype(np.float32)
                         logger.debug(f"Column '{col}' converted to {np.float32}.")
                 except (TypeError, ValueError) as te:
                     logger.warning(f"TypeError/ValueError during float range comparison for column '{col}'. Keeping {df_opt[col].dtype}. Error: {te}")

        # Object to Category conversion
        elif col_type == 'object':
            if len(df_opt[col]) > 0 and df_opt[col].dtype != 'category':
                try:
                    num_unique = df_opt[col].nunique()
                    # Use the passed config dictionary HERE
                    category_threshold = config.get("category_conversion_threshold", 0.5)
                    if num_unique / len(df_opt[col]) < category_threshold:
                        df_opt[col] = df_opt[col].astype('category')
                        logger.debug(f"Column '{col}' converted to category.")
                except Exception as e: # Handle potential errors during nunique() or astype()
                    # Add error context to the warning
                    logger.warning(f"Could not check uniqueness or convert object column '{col}' to category. Error: {e}", exc_info=True) # Add exc_info

    end_mem = df_opt.memory_usage(deep=True).sum() / 1024**2
    mem_change_msg = f"Memory usage: {end_mem:.2f} MB."
    if start_mem > 0 and start_mem > end_mem: # Only report reduction
        mem_reduction_pct = (start_mem - end_mem) / start_mem * 100
        mem_change_msg = f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({mem_reduction_pct:.1f}% reduction)."
    elif start_mem > 0 and end_mem > start_mem:
        mem_increase_pct = (end_mem - start_mem) / start_mem * 100
        mem_change_msg = f"Memory usage increased from {start_mem:.2f} MB to {end_mem:.2f} MB ({mem_increase_pct:.1f}% increase)." # e.g., due to nullable ints

    logger.info(mem_change_msg)
    return df_opt

# --- AutoML Runner Class ---
class AutoMLRunner:
    """
    Encapsulates DISCRETE AutoML workflow steps using PyCaret and MLflow,
    designed to be called sequentially by an external controller (e.g., API service).
    Incorporates enhanced features like profiling, drift check, explanations, etc.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the runner with configuration.
        (Initializes directories, base config, transient state)
        """
        self.config = config
        logger.debug(f"Initializing AutoMLRunner with config: { {k:v for k,v in config.items() if k not in ['data','feature_columns','setup_params_extra']} }") # Avoid logging large data/lists

        # --- Validate essential config keys needed immediately ---
        required_on_init = ["session_id", "output_base_dir", "mlflow_tracking_uri", "experiment_name"]
        missing_keys = [k for k in required_on_init if k not in config or not config[k]]
        if missing_keys:
             msg = f"AutoMLRunner config missing required keys: {', '.join(missing_keys)}"
             logger.error(msg)
             raise ValueError(msg)

        self.session_id = config["session_id"]
        self.output_base_dir = config["output_base_dir"] # Assumed absolute path

        # Define session-specific paths
        self.session_dir = os.path.join(self.output_base_dir, f"automl_{self.session_id}")
        self.model_save_dir = os.path.join(self.session_dir, "models")
        self.plot_save_dir = os.path.join(self.session_dir, "plots")
        self.experiment_save_dir = os.path.join(self.session_dir, "experiments")
        self.report_save_dir = os.path.join(self.session_dir, "reports") # for reports
        self.last_feature_importance_plot_path: Optional[str] = None # Store plot path
        self.last_tuned_cv_results_df: Optional[pd.DataFrame] = None # Store tuning CV results
        self.last_tuned_best_params: Optional[Dict] = None # Store best params

        # --- Create directories idempotently ---
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            os.makedirs(self.model_save_dir, exist_ok=True)
            os.makedirs(self.plot_save_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(self.report_save_dir, exist_ok=True) # NEW
            logger.info(f"Initialized Runner for Session ID: {self.session_id}")
            logger.info(f" -> Artifact Base Directory: {self.session_dir}")
        except OSError as e:
            logger.error(f"Failed to create session directories in '{self.output_base_dir}': {e}", exc_info=True)
            raise RuntimeError(f"Directory creation failed for session {self.session_id}") from e

        # --- Transient State Variables (Includes new ones) ---
        self.data: Optional[pd.DataFrame] = None # Holds data loaded in Step 1
        self.train_data: Optional[pd.DataFrame] = None # Data used for training (post-split/sampling)
        self.test_data: Optional[pd.DataFrame] = None # Hold-out data from setup
        self.task_type: Optional[str] = self.config.get("task_type")
        self.pycaret_module: Optional[Any] = None
        self.sort_metric: Optional[str] = None
        self.optimize_metric: Optional[str] = None
        self.setup_env: Optional[Any] = None # Holds result of setup() within step 1
        self.preprocessor: Optional[Any] = None # NEW: Fitted preprocessing pipeline from setup
        self.results_df: Optional[pd.DataFrame] = None # Holds result of compare_models() within step 1
        self.tuned_best_model: Optional[Any] = None # Holds result of tune_model() within step 2
        self.final_model: Optional[Any] = None # Holds result of finalize_model() within step 3
        self.latest_analysis_metrics: Optional[Dict] = None # Store metrics from analysis for model card
        self.training_data_profile_path: Optional[str] = None # Path to data profile report

        self._setup_mlflow() # Setup MLflow tracking details

    # ------------------- Internal Helper Methods -------------------

    # --- MLflow Run Cleanup Helper ---
    def _ensure_no_active_mlflow_run(self, step_name: str = "Unknown"):
        """Aggressively ends any unexpected active MLflow runs before starting a step."""
        retry_count = 0
        max_retries = 5
        while mlflow.active_run() is not None and retry_count < max_retries:
            active_run_obj = mlflow.active_run()
            active_run_id = active_run_obj.info.run_id
            logger.warning(f"[{step_name}-Cleanup-{retry_count}] Detected active run {active_run_id} before starting step. Attempting to end it.")
            try:
                mlflow.end_run(status="KILLED") # Mark as killed if ended unexpectedly
                logger.info(f"[{step_name}-Cleanup-{retry_count}] Ended run {active_run_id} with status KILLED.")
            except Exception as e:
                logger.error(f"[{step_name}-Cleanup-{retry_count}] Failed to end active run {active_run_id}: {e}. Retrying check.")
            retry_count += 1
            time.sleep(0.1)

        if mlflow.active_run() is not None:
            final_active_run_id = mlflow.active_run().info.run_id
            msg = f"[{step_name}] Failed to clear active MLflow run ({final_active_run_id}) after {max_retries} retries. Cannot start step."
            logger.error(msg)
            raise RuntimeError(msg)

    def _setup_mlflow(self):
        """(Internal Helper) Sets up MLflow tracking URI and experiment."""
        try:
            logger.info("Setting up MLflow Tracking...")
            mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])
            mlflow.set_experiment(self.config["experiment_name"])
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            logger.info(f"MLflow experiment set to: {self.config['experiment_name']}")
            # Ensure no runs are active during runner instance initialization
            self._ensure_no_active_mlflow_run("RunnerInit")
        except Exception as e:
            logger.error(f"Failed to setup MLflow tracking: {e}", exc_info=True)
            raise RuntimeError("MLflow setup failed") from e # Make it fatal

    # --- UPDATED: _detect_task_type (Keep robust version from Code 1) ---
    def _detect_task_type(self) -> bool:
        """
        (Internal Helper) Auto-Detects Task Type based on self.data.
        Improved logic to better distinguish numeric types.
        """
        # --- Pre-checks ---
        required_keys = ["target_column", "unique_value_threshold_for_classification"]
        missing = [k for k in required_keys if k not in self.config]
        if missing:
            logger.error(f"Cannot detect task type. Config missing keys: {missing}")
            return False
        if self.data is None:
            logger.error("Cannot detect task type. Data not loaded (self.data is None).")
            return False
        target_col = self.config["target_column"]
        if target_col not in self.data.columns:
            logger.error(f"Cannot detect task type. Target column '{target_col}' not found in loaded data.")
            return False
        # --- End Pre-checks ---

        logger.info("Auto-Detecting Task Type (Improved Logic)...")
        target_series = self.data[target_col]
        n_unique = target_series.nunique()
        dtype = target_series.dtype
        threshold = self.config["unique_value_threshold_for_classification"] # Threshold primarily for integers now

        # --- Detection Logic ---
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            detected_type = 'classification'
        elif pd.api.types.is_bool_dtype(dtype):
            detected_type = 'classification'
        elif pd.api.types.is_float_dtype(dtype):
            float_classification_max_unique = self.config.get("float_classification_max_unique", 5) # Default to low value
            if n_unique <= float_classification_max_unique:
                detected_type = 'classification'
                logger.warning(f"Detected task type: Classification (Target is float with only {n_unique} unique values <= threshold {float_classification_max_unique}). Verify this is intended.")
            else:
                detected_type = 'regression'
        elif pd.api.types.is_integer_dtype(dtype):
            is_binary_01 = (n_unique == 2 and target_series.isin([0, 1]).all() and not target_series.isnull().any())
            if is_binary_01:
                detected_type = 'classification'
                logger.info("Detected task type: Binary Classification (0/1 Integer Target)")
            elif n_unique < threshold:
                detected_type = 'classification'
            else:
                detected_type = 'regression'
        else:
            logger.warning(f"Could not reliably determine task type for target dtype {dtype}. Defaulting to classification.")
            detected_type = 'classification' # Fallback

        logger.info(f"Final Detected Task Type: {detected_type.upper()}")
        self.task_type = detected_type
        self.config['task_type'] = self.task_type # Update config state
        return True

    # --- _setup_pycaret (Integrate setup_params_extra) ---
    def _setup_pycaret(self) -> bool:
        """(Internal Helper) Sets up PyCaret Environment based on self.data and self.task_type."""
        if self.data is None or self.task_type is None:
            logger.error("Data or task type not available for PyCaret setup.")
            return False
        if not all(k in self.config for k in ["target_column", "session_id", "experiment_name", "numeric_imputation"]):
             logger.error("Cannot setup PyCaret. Config missing essential setup keys.")
             return False

        logger.info("Setting up PyCaret Environment on prepared data...")
        start_time = time.time()

        target_col = self.config["target_column"]
        feature_cols_from_config = self.config.get("feature_columns")

        ignore_features_list = None
        if isinstance(feature_cols_from_config, list) and feature_cols_from_config:
            all_columns = set(self.data.columns)
            valid_feature_cols = set(feature_cols_from_config)
            target_set = {target_col}
            missing_features = list(valid_feature_cols - all_columns)
            if missing_features:
                 logger.error(f"Config Error: Requested feature_columns not found: {missing_features}. Aborting setup.")
                 return False
            ignore_features_list = list(all_columns - valid_feature_cols - target_set)
            logger.info(f"Using specified feature_columns. Ignoring {len(ignore_features_list)} other columns.")
            if ignore_features_list: logger.debug(f"Columns to ignore: {ignore_features_list}")
            if target_col in ignore_features_list:
                logger.error(f"Config Error: Target column '{target_col}' in ignore list. Aborting setup.")
                return False
        else:
            logger.info("Using all columns as features (excluding target) as 'feature_columns' not specified.")

        # --- Base Setup Params ---
        setup_params = {
            "data": self.data,
            "target": self.config["target_column"],
            "session_id": self.session_id,
            "log_experiment": True, # Log to the active MLflow run
            "experiment_name": self.config["experiment_name"],
            "numeric_imputation": self.config.get("numeric_imputation", "mean"),
            "categorical_imputation": self.config.get("categorical_imputation", "mode"),
            "n_jobs": self.config.get("n_jobs", -1),
            "fold_strategy": self.config.get("fold_strategy", "stratifiedkfold" if self.task_type == "classification" else "kfold"),
            "fold": self.config.get("baseline_folds", 5),
            "html": self.config.get("pycaret_setup_html", False),
            "verbose": self.config.get("pycaret_setup_verbose", False),
            "use_gpu": self.config.get("use_gpu", False),
            # --- Add extra setup params from config ---
            **self.config.get("setup_params_extra", {})
        }

        if ignore_features_list is not None:
             # Prevent setup_params_extra from overriding calculated list
             if "ignore_features" in setup_params and setup_params["ignore_features"] != ignore_features_list:
                 logger.warning("Config 'setup_params_extra' specified 'ignore_features', but it was already determined by 'feature_columns'. Using calculated list.")
             setup_params["ignore_features"] = ignore_features_list

        # --- Task-Specific Setup ---
        if self.task_type == 'classification':
            if not all(k in self.config for k in ["sort_metric_classification", "optimize_metric_classification"]):
                 logger.error("Config missing classification sort/optimize metrics.")
                 return False
            self.pycaret_module = pyclf
            self.sort_metric = self.config["sort_metric_classification"]
            self.optimize_metric = self.config["optimize_metric_classification"]
        elif self.task_type == 'regression':
            if not all(k in self.config for k in ["sort_metric_regression", "optimize_metric_regression"]):
                 logger.error("Config missing regression sort/optimize metrics.")
                 return False
            self.pycaret_module = pyreg
            self.sort_metric = self.config["sort_metric_regression"]
            self.optimize_metric = self.config["optimize_metric_regression"]
            setup_params["fold_strategy"] = "kfold"
            # Apply regression-specific setup defaults if not overridden
            setup_params.setdefault("normalize", True) # Example default
        else:
            logger.error(f"Invalid task type during setup: {self.task_type}")
            return False

        logger.info(f"Using PyCaret Module: {self.pycaret_module.__name__}")
        logger.info(f"Sort Metric: {self.sort_metric}, Optimize Metric: {self.optimize_metric}")
        logger.info(f"PyCaret Setup Parameters (subset): n_jobs={setup_params.get('n_jobs', 'Default')}, ignore_features_count={len(setup_params.get('ignore_features', []))}, extra_params_keys={list(self.config.get('setup_params_extra', {}).keys())}")

        try:
            self.setup_env = self.pycaret_module.setup(**setup_params)
            setup_duration = time.time() - start_time
            logger.info(f"PyCaret setup completed in {setup_duration:.2f} seconds.")

            # --- Extract preprocessor and save train/test data ---
            self.preprocessor = self.setup_env.pipeline # Get the fitted preprocessor
            self.train_data = self.setup_env.X_train.join(self.setup_env.y_train)
            self.test_data = self.setup_env.X_test.join(self.setup_env.y_test)
            logger.info(f"Preprocessor extracted. Train shape: {self.train_data.shape}, Test shape: {self.test_data.shape}")

            # Log key setup params explicitly to MLflow if run is active
            if mlflow.active_run():
                mlflow.log_param("pycaret_manual_sampling_applied", self.config.get("use_sampling_in_setup", False))
                mlflow.log_param("pycaret_setup_input_rows", self.data.shape[0])
                mlflow.log_param("pycaret_n_jobs", setup_params.get('n_jobs', 'Default'))
                mlflow.log_param("pycaret_fold_strategy", setup_params.get('fold_strategy'))
                mlflow.log_param("pycaret_baseline_folds", setup_params.get('fold'))
                mlflow.log_param("pycaret_ignored_features_count", len(setup_params.get("ignore_features", [])))
                # Log extra params used
                mlflow.log_dict(self.config.get("setup_params_extra", {}), "pycaret_setup_params_extra.json")
                mlflow.log_metric("pycaret_setup_duration_sec", setup_duration)
            return True
        except Exception as e:
            logger.error(f"PyCaret setup failed: {e}", exc_info=True)
            if mlflow.active_run():
                mlflow.log_param("pycaret_setup_error", f"{type(e).__name__}: {str(e)[:500]}")
            return False

    # --- _compare_models ---
    def _compare_models(self) -> bool:
        """(Internal Helper) Compares models using PyCaret."""
        if self.pycaret_module is None or self.sort_metric is None:
            logger.error("PyCaret module/sort_metric not set. Cannot compare models.")
            return False
        logger.info("Comparing Models...")
        start_time = time.time()

        compare_params = {
            "sort": self.sort_metric,
            "fold": self.config.get("baseline_folds", 5),
            "include": self.config.get("include_models_compare", None),
            "exclude": self.config.get("exclude_models_compare", None),
            "n_select": self.config.get("compare_n_select", 1),
            "verbose": self.config.get("pycaret_compare_verbose", False)
        }
        compare_params = {k: v for k, v in compare_params.items() if v is not None}
        logger.info(f"Compare Models Params: {compare_params}")

        try:
            _ = self.pycaret_module.compare_models(**compare_params)
            self.results_df = self.pycaret_module.pull()

            if self.results_df is None or self.results_df.empty:
                 logger.warning("compare_models did not return a results dataframe.")
            elif mlflow.active_run():
                logger.info("\nTop Model Comparison Results (Cross-Validated):")
                try:
                    results_html = self.results_df.to_html(escape=False, max_rows=20)
                    mlflow.log_text(results_html, "compare_models_results.html")
                    if self.sort_metric in self.results_df.columns:
                        top_metric_value = self.results_df.iloc[0][self.sort_metric]
                        if isinstance(top_metric_value, (int, float, np.number)):
                             mlflow.log_metric(f"compare_top1_cv_{self.sort_metric}", float(top_metric_value))
                             logger.info(f"Top model CV {self.sort_metric}: {top_metric_value:.4f}")
                        else:
                             logger.warning(f"Top model metric '{self.sort_metric}' has non-numeric type: {type(top_metric_value)}")
                    else:
                        logger.warning(f"Sort metric '{self.sort_metric}' not found in results.")
                except Exception as log_err:
                     logger.error(f"Failed to log compare_models results: {log_err}", exc_info=True)

            compare_duration = time.time() - start_time
            logger.info(f"Model comparison completed in {compare_duration:.2f} seconds.")
            if mlflow.active_run():
                mlflow.log_metric("compare_models_duration_sec", compare_duration)
            return True

        except Exception as e:
            logger.error(f"Model comparison failed: {e}", exc_info=True)
            if mlflow.active_run():
                mlflow.log_param("compare_error", f"{type(e).__name__}: {str(e)[:500]}")
            return False

    # --- _analyze_baseline_models ---
    def _analyze_baseline_models(self) -> bool:
        """(Internal Helper) Creates, evaluates, logs plots for specified baseline models."""
        if self.pycaret_module is None or self.task_type is None:
             logger.error("PyCaret module/task_type not set. Cannot analyze baselines.")
             return False

        baseline_ids_key = f"baseline_{self.task_type}_models"
        baseline_model_ids = self.config.get(baseline_ids_key, [])

        if not isinstance(baseline_model_ids, list) or not baseline_model_ids:
             logger.info(f"No baseline models specified for task '{self.task_type}' via key '{baseline_ids_key}'. Skipping.")
             return True

        logger.info(f"--- Analyzing Baseline Models: {baseline_model_ids} ---")
        success_count = 0
        parent_run = mlflow.active_run()
        parent_run_id = parent_run.info.run_id if parent_run else None
        baseline_analysis_start_time = time.time()

        for baseline_id in baseline_model_ids:
            if not isinstance(baseline_id, str):
                 logger.warning(f"Skipping invalid baseline ID: {baseline_id} (must be string).")
                 continue
            logger.info(f"Processing baseline model: {baseline_id}")
            run_name = f"Baseline_{baseline_id}_{self.session_id}"
            # Use nested run for isolated logging
            with mlflow.start_run(run_name=run_name, nested=True) as baseline_run:
                current_run_id = baseline_run.info.run_id
                logger.debug(f"Starting nested run for baseline {baseline_id}: {current_run_id}")
                if parent_run_id: mlflow.log_param("parent_run_id", parent_run_id)
                mlflow.set_tags({"model_stage": "baseline", "model_id": baseline_id,
                                 "session_id": str(self.session_id), "baseline_status": "started"})
                start_time_model = time.time()
                baseline_model_obj = None

                try:
                    # Create baseline model (uses CV folds specified in setup)
                    baseline_model_obj = self.pycaret_module.create_model(
                        baseline_id,
                        fold=self.config.get("baseline_folds", 5),
                        verbose=False
                    )
                    if baseline_model_obj is None:
                        raise RuntimeError(f"create_model returned None for '{baseline_id}'.")
                    logger.info(f"Created baseline '{baseline_id}' in {time.time() - start_time_model:.2f}s")

                    # Pull & Log baseline CV metrics
                    baseline_metrics_df = self.pycaret_module.pull()
                    if baseline_metrics_df is not None and not baseline_metrics_df.empty:
                        if 'Mean' in baseline_metrics_df.index:
                            mean_metrics = baseline_metrics_df.loc['Mean'].to_dict()
                            mean_metrics_logged = {f"cv_mean_{k}": v for k, v in mean_metrics.items() if isinstance(v, (int, float, np.number))}
                            if mean_metrics_logged:
                                 mlflow.log_metrics(mean_metrics_logged)
                            else: logger.warning(f"No numeric mean CV metrics for baseline '{baseline_id}'.")
                        else: logger.warning(f"'Mean' row not found in CV metrics for baseline '{baseline_id}'.")
                        try: # Log results table
                            metrics_html = baseline_metrics_df.to_html(escape=False, max_rows=20)
                            mlflow.log_text(metrics_html, f"baseline_{baseline_id}_cv_metrics.html")
                        except Exception as log_err: logger.error(f"Failed log baseline metrics table for {baseline_id}: {log_err}")
                    else: logger.warning(f"Could not pull CV metrics for baseline '{baseline_id}'.")

                    # Log feature importance plot if applicable (using refined check from Code 1)
                    estimator_for_plot = baseline_model_obj
                    can_plot_feature = False
                    if hasattr(estimator_for_plot, 'steps'):
                        final_step = estimator_for_plot.steps[-1][1]
                        if hasattr(final_step, 'feature_importances_') or hasattr(final_step, 'coef_'): can_plot_feature = True
                    elif hasattr(estimator_for_plot, 'feature_importances_') or hasattr(estimator_for_plot, 'coef_'): can_plot_feature = True

                    if can_plot_feature:
                        self._log_plot_artifact(
                            plot_function=lambda save, verbose: self.pycaret_module.plot_model(baseline_model_obj, plot='feature', save=save, verbose=verbose),
                            plot_name="Feature_Importance", model_id=baseline_id, model_stage="baseline"
                        )
                    else:
                        logger.debug(f"Skipping feature importance plot for baseline '{baseline_id}' (not applicable).")

                    # Save baseline model artifact (optional)
                    if self.config.get("save_baseline_models", False):
                         save_path_base = self._save_model_artifact(baseline_model_obj, "baseline", baseline_id)
                         if not save_path_base: logger.warning(f"Failed to save baseline artifact for {baseline_id}")

                    model_duration = time.time() - start_time_model
                    mlflow.log_metric("model_duration_seconds", model_duration)
                    mlflow.set_tag("baseline_status", "completed")
                    success_count += 1

                except Exception as e:
                    logger.error(f"Error processing baseline '{baseline_id}': {e}", exc_info=True)
                    try: # Ensure status marked failed even if logging fails
                         mlflow.set_tag("baseline_status", "failed")
                         mlflow.log_param("error", f"{type(e).__name__}: {str(e)[:500]}")
                         if start_time_model: mlflow.log_metric("model_duration_seconds", time.time() - start_time_model)
                    except Exception as log_fail_err:
                         logger.error(f"Failed log error details for baseline {baseline_id} to MLflow: {log_fail_err}")

        total_baseline_duration = time.time() - baseline_analysis_start_time
        logger.info(f"Baseline analysis finished. Processed {success_count}/{len(baseline_model_ids)} models in {total_baseline_duration:.2f}s.")
        # Log overall baseline duration to parent run if active
        if parent_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == parent_run_id:
             try:
                 mlflow.log_metric("analyze_baseline_duration_sec", total_baseline_duration)
                 mlflow.log_metric("baseline_models_success_count", success_count)
             except Exception as log_err:
                  logger.error(f"Failed log overall baseline metrics to parent run {parent_run_id}: {log_err}")

        return True # Return True if process ran, even if some models failed


    # --- _analyze_tuned_model (Integrate SHAP option from config) ---
    def _analyze_tuned_model(self) -> bool:
        """(Internal Helper) Generates analysis plots for the tuned model."""
        if self.tuned_best_model is None or self.pycaret_module is None or self.task_type is None:
            logger.error("Tuned model/PyCaret module/task_type not available for analysis.")
            return False
        
        self.last_feature_importance_plot_path = None
        self.latest_analysis_metrics = None #
        tuned_model_id = self._get_model_id(self.tuned_best_model) or "tuned_model"
        logger.info(f"--- Analyzing Tuned Model: {tuned_model_id} ({self.task_type}) ---")
        analysis_start_time = time.time()
        plots_logged_count = 0
        holdout_metrics = {} # Store holdout metrics here

        try:
            logger.info("Generating analysis plots for tuned model (using hold-out set)...")

         
            # predict_model implicitly uses hold-out set after tuning
            hold_out_predictions = self.pycaret_module.predict_model(self.tuned_best_model, verbose=False)
            hold_out_metrics_df = self.pycaret_module.pull() # metrics from predict_model evaluation

            if hold_out_metrics_df is not None and not hold_out_metrics_df.empty:
                # Store metrics for potential use in model card
                holdout_metrics = hold_out_metrics_df.iloc[0].to_dict()
                self.latest_analysis_metrics = holdout_metrics # Save to instance variable
                logger.info(f"Tuned Model Hold-out Metrics:\n{pd.Series(holdout_metrics).to_string()}")
                # Log numeric metrics to MLflow
                metrics_to_log = {f"tuned_holdout_{k}": v for k,v in holdout_metrics.items() if isinstance(v, (int, float, np.number))}
                if metrics_to_log and mlflow.active_run(): mlflow.log_metrics(metrics_to_log)
                # Log table as artifact
                if mlflow.active_run():
                     try:
                         metrics_html = hold_out_metrics_df.to_html(escape=False)
                         mlflow.log_text(metrics_html, f"tuned_{tuned_model_id}_holdout_metrics.html")
                     except Exception as log_err: logger.error(f"Failed log tuned holdout metrics table: {log_err}")
            else:
                logger.warning(f"Could not pull hold-out metrics for tuned model '{tuned_model_id}'.")
                self.latest_analysis_metrics = None


            # --- Log Feature Importance ---
            estimator_for_plot = self.tuned_best_model
            can_plot_feature = False
            if hasattr(estimator_for_plot, 'steps'):
                final_step = estimator_for_plot.steps[-1][1]
                if hasattr(final_step, 'feature_importances_') or hasattr(final_step, 'coef_'): can_plot_feature = True
            elif hasattr(estimator_for_plot, 'feature_importances_') or hasattr(estimator_for_plot, 'coef_'): can_plot_feature = True

            if can_plot_feature:
                if self._log_plot_artifact(
                    plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='feature', save=save, verbose=verbose),
                    plot_name="Feature_Importance", model_id=tuned_model_id, model_stage="tuned"
                    ): plots_logged_count +=1
            else:
                 logger.debug(f"Skipping feature importance plot for tuned '{tuned_model_id}' (not applicable).")


            # --- Log task-specific plots ---
            plots_to_generate = []
            if self.task_type == 'classification':
                 plots_to_generate = self.config.get("classification_analysis_plots", ['auc', 'confusion_matrix', 'pr', 'class_report'])
            elif self.task_type == 'regression':
                  plots_to_generate = self.config.get("regression_analysis_plots",['residuals', 'error', 'cooks'])

            for plot_type in plots_to_generate:
                 if self._log_plot_artifact(
                     plot_function=lambda save, verbose, p=plot_type: self.pycaret_module.plot_model(self.tuned_best_model, plot=p, save=save, verbose=verbose),
                     plot_name=plot_type.replace('_',' ').title(), model_id=tuned_model_id, model_stage="tuned"
                     ): plots_logged_count += 1

            # --- Conditional SHAP plot ---
            if self.config.get("shap_enabled_if_possible", False):
                actual_estimator_id = self._get_model_id(self.tuned_best_model)
                # PyCaret's interpret_model has limited support, check known compatible models
                supported_shap_models = self.config.get("shap_supported_model_ids_interpret",
                                                        {'et', 'lightgbm', 'xgboost', 'rf', 'dt', 'catboost', 'knn', 'ridge', 'lr', 'svm', 'par', 'huber'})

                if shap and actual_estimator_id and actual_estimator_id in supported_shap_models:
                    logger.info(f"Attempting SHAP summary plot for '{actual_estimator_id}' via interpret_model...")
                    if self._log_plot_artifact(
                        plot_function=lambda save: self.pycaret_module.interpret_model(self.tuned_best_model, plot='summary', save=save),
                        plot_name="SHAP_Summary", model_id=tuned_model_id, model_stage="tuned"
                        ): plots_logged_count += 1
                elif shap and actual_estimator_id:
                    logger.warning(f"SHAP summary plot via interpret_model() may not be supported or is skipped for model type '{actual_estimator_id}'. Consider using `explain_prediction` for instance-level SHAP.")
                elif not shap:
                     logger.warning("SHAP library not installed, skipping SHAP plot generation.")
                else:
                     logger.warning("Could not determine actual estimator ID for SHAP plot.")
            else:
                 logger.info("SHAP plot generation skipped based on config ('shap_enabled_if_possible').")


            analysis_duration = time.time() - analysis_start_time
            logger.info(f"Tuned model analysis completed in {analysis_duration:.2f} seconds. Logged {plots_logged_count} plots.")
            if mlflow.active_run():
                 mlflow.log_metric("analyze_tuned_duration_sec", analysis_duration)
                 mlflow.log_metric("tuned_plots_logged_count", plots_logged_count)

            return True # Return True even if some plots failed

        except Exception as e:
            logger.error(f"Failed during tuned model analysis: {e}", exc_info=True)
            if mlflow.active_run():
                mlflow.log_param("tuned_analysis_error", f"{type(e).__name__}: {str(e)[:500]}")
            return False

    # ---_log_plot_artifact ---
    def _log_plot_artifact(self, plot_function: callable, plot_name: str, model_id: str, model_stage: str, **kwargs) -> bool:
        """
        (Internal Helper) Generates, saves LOCALLY to session folder, renames, and logs a plot artifact.
        Uses sanitize_filename and handles interpret_model saving.
        Returns True if plot logging to MLflow was successful, False otherwise.
        """
        logger.info(f"Attempting to generate plot: '{plot_name}' for {model_stage} model '{model_id}'")
        original_save_path = None
        safe_plot_name = sanitize_filename(plot_name) # Use new helper
        safe_model_id = sanitize_filename(model_id) # Sanitize model id too
        final_artifact_filename = f"{model_stage}_{safe_model_id}_{safe_plot_name}.png"
        # Save plots relative to session plot directory
        final_local_path = os.path.join(self.plot_save_dir, final_artifact_filename)
        mlflow_logged = False

        try:
            os.makedirs(self.plot_save_dir, exist_ok=True) # Ensure dir exists

            # --- Prepare kwargs for plot function ---
            func_sig = inspect.signature(plot_function)
            plot_kwargs = kwargs.copy()
            if 'save' in func_sig.parameters: plot_kwargs['save'] = True
            if 'verbose' in func_sig.parameters: plot_kwargs['verbose'] = False
            else: plot_kwargs.pop('verbose', None) # Remove if not accepted

            # --- Execute plot function (Handle interpret_model separately) ---
            is_interpret_call = 'interpret_model' in str(plot_function)

            if is_interpret_call:
                logger.debug(f"Calling interpret_model for plot '{plot_name}' - attempting manual save.")
                try:
                    backend = plt.get_backend()
                    plt.switch_backend('Agg')
                    plot_function(**plot_kwargs)
                    plt.savefig(final_local_path, bbox_inches='tight', format='png')
                    plt.close()
                    original_save_path = final_local_path # Path where it was saved directly
                    logger.info(f"Manually saved interpret_model plot to: {final_local_path}")
                    plt.switch_backend(backend)
                except Exception as interpret_save_err:
                      logger.warning(f"Failed to manually save interpret_model plot '{plot_name}': {interpret_save_err}", exc_info=True)
                      original_save_path = None
                      try: plt.switch_backend(backend) # Ensure backend restored
                      except: pass # Ignore errors during backend restore
            else:
                 # Standard plot_model call - returns the path where it saved
                 original_save_path = plot_function(**plot_kwargs)

            # --- Move and Log if plot file exists ---
            if original_save_path and os.path.exists(original_save_path):
                # If PyCaret saved it somewhere else (or directly to target path), check and move if needed
                if os.path.abspath(original_save_path) != os.path.abspath(final_local_path):
                    logger.debug(f"PyCaret saved plot temporarily to: {original_save_path}")
                    try:
                        # Ensure target directory exists before moving
                        os.makedirs(os.path.dirname(final_local_path), exist_ok=True)
                        shutil.move(original_save_path, final_local_path)
                        logger.info(f"Moved plot file to session plots folder: {final_local_path}")
                    except OSError as move_e:
                        logger.error(f"Could not move plot file from '{original_save_path}' to '{final_local_path}': {move_e}", exc_info=True)
                        # Fallback: try logging the original if move failed
                        if os.path.exists(original_save_path):
                            final_local_path = original_save_path
                            logger.warning(f"Move failed. Attempting to log original plot artifact: {original_save_path}")
                        else: final_local_path = None

                # --- Log the artifact to MLflow ---
                if final_local_path and os.path.exists(final_local_path):
                    if mlflow.active_run():
                        try:
                            # Use "plots" as the artifact subdirectory in MLflow
                            mlflow.log_artifact(final_local_path, artifact_path="plots")
                            logger.info(f"Successfully logged plot artifact: 'plots/{final_artifact_filename}'")
                            mlflow_logged = True
                        except Exception as log_e:
                            logger.error(f"Could not log plot artifact '{final_local_path}': {log_e}", exc_info=True)
                    else: 
                        logger.warning(f"No active MLflow run. Cannot log plot artifact '{final_artifact_filename}'.")

                    if plot_name == "Feature_Importance" and final_local_path:
                        self.last_feature_importance_plot_path = final_local_path
                        logger.info(f"Stored feature importance plot path: {final_local_path}")
                    
                elif final_local_path is None:
                     logger.error(f"Plot move failed and original path missing. Cannot log artifact '{final_artifact_filename}'.")

            else:
                logger.warning(f"Plot file path invalid, not generated, or not saved for '{plot_name}' (Model: {model_stage} {model_id}). Expected path: {original_save_path}")

        except ImportError as imp_err:
             logger.warning(f"Could not generate plot '{plot_name}'. Required library missing: {imp_err}")
        except TypeError as te:
             logger.warning(f"Could not generate plot '{plot_name}' for {model_stage} model '{model_id}'. Type might not support plot or params incorrect. Error: {te}")
        except Exception as plot_e:
            logger.error(f"Failed during plot generation/saving/logging for '{plot_name}': {plot_e}", exc_info=True)

        return mlflow_logged

    # --- _save_model_artifact (Ensure metadata includes task_type) ---
    def _save_model_artifact(self, model_object: Any, model_stage: str, model_id: str) -> Optional[str]:
        """
        (Internal Helper) Saves model artifact and metadata (incl task_type).
        Logs paths and artifacts to MLflow if active. Sanitize names.
        Returns the base path (without extension) if successful, None otherwise.
        """
        if self.pycaret_module is None or self.task_type is None:
             logger.error(f"Cannot save model artifact for {model_stage}/{model_id} - PyCaret module or task_type unknown.")
             return None
        if not model_object:
             logger.error(f"Cannot save model artifact - model object for {model_stage}/{model_id} is None.")
             return None

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_id = sanitize_filename(str(model_id)) # Sanitize ID
            model_filename_base = f"{model_stage}_{safe_model_id}_{timestamp}_{self.session_id}"
            save_path_base = os.path.join(self.model_save_dir, model_filename_base)
            os.makedirs(self.model_save_dir, exist_ok=True)

            # --- Save Model using PyCaret ---
            self.pycaret_module.save_model(model_object, save_path_base, verbose=False)
            full_save_path_pkl = f"{save_path_base}.pkl"

            if not os.path.exists(full_save_path_pkl):
                 logger.error(f"PyCaret's save_model failed silently for {model_stage}/{model_id}.")
                 raise RuntimeError(f"save_model did not create the expected file: {full_save_path_pkl}")

            # --- Save Metadata (CRITICAL: Include task_type) ---
            metadata_path = f"{save_path_base}_meta.json"
            metadata = {
                "task_type": self.task_type, # Ensure task_type is saved
                "pycaret_version": pycaret.__version__,
                "model_stage": model_stage,
                "model_id_used": model_id,
                "session_id": self.session_id,
                "timestamp": timestamp
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Model '{model_id}' ({model_stage}) saved locally to: {full_save_path_pkl}")
            logger.info(f"Metadata saved locally to: {metadata_path}")

            # --- Log to MLflow if active ---
            if mlflow.active_run():
                try:
                    mlflow.log_param(f"{model_stage}_{safe_model_id}_local_save_path_pkl", full_save_path_pkl)
                    mlflow.log_param(f"{model_stage}_{safe_model_id}_local_save_path_meta", metadata_path)
                    # Log both metadata and model PKL as artifacts under a structured path
                    artifact_subpath = f"models/{model_stage}"
                    mlflow.log_artifact(metadata_path, artifact_path=artifact_subpath)
                    mlflow.log_artifact(full_save_path_pkl, artifact_path=artifact_subpath)
                    logger.info(f"Logged model and metadata artifacts to MLflow: {artifact_subpath}/")

                except Exception as log_e:
                    logger.error(f"Failed to log model/metadata artifacts to MLflow for {model_stage}/{model_id}: {log_e}", exc_info=True)
            else:
                logger.warning("No active MLflow run. Model/metadata artifacts not logged to MLflow.")

            return save_path_base # Return base path without extension

        except Exception as e:
            logger.error(f"Failed to save model artifact {model_stage} {model_id}: {e}", exc_info=True)
            return None

    # --- _load_model_and_metadata (Adopted from Code 2 for prediction/explanation) ---
    def _load_model_and_metadata(self, model_base_path: str) -> Tuple[Optional[Any], Optional[str]]:
        """
        Loads model .pkl and associated _meta.json based on base path.
        Determines task_type from metadata for correct loading.
        Returns (model_object, task_type) or (None, None) on failure.
        """
        model_pkl_path = f"{model_base_path}.pkl"
        model_meta_path = f"{model_base_path}_meta.json"
        loaded_model = None
        task_type = None # Crucial variable

        # --- Check file existence ---
        if not os.path.exists(model_pkl_path):
            logger.error(f"Model file not found: {model_pkl_path}")
            return None, None
        if not os.path.exists(model_meta_path):
            logger.error(f"Metadata file not found: {model_meta_path}. Cannot determine task type for loading.")
            return None, None # Fail if no metadata

        # --- Load Metadata to get Task Type ---
        try:
            with open(model_meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            task_type = metadata.get("task_type")
            if not task_type:
                logger.error(f"Task type not found or empty in metadata file: {model_meta_path}")
                return None, None
            logger.info(f"Loaded task type '{task_type}' from metadata.")
        except Exception as e:
            logger.error(f"Error loading or parsing metadata file {model_meta_path}: {e}", exc_info=True)
            return None, None

        # --- Determine PyCaret module and Load Model ---
        load_module = None
        if task_type == 'classification': load_module = pyclf
        elif task_type == 'regression': load_module = pyreg
        else:
            logger.error(f"Invalid task_type '{task_type}' found in metadata. Cannot load.")
            return None, None

        try:
            logger.info(f"Loading model using {load_module.__name__} based on metadata...")
            loaded_model = load_module.load_model(model_base_path, verbose=False)
            if loaded_model is None:
                 raise RuntimeError("load_model returned None unexpectedly.")
            logger.info(f"Model loaded successfully from {model_base_path}.pkl")
            return loaded_model, task_type # Return both
        except Exception as e_load:
            logger.error(f"Failed loading model {model_base_path}.pkl with {load_module.__name__}: {e_load}", exc_info=True)
            return None, None

    def _get_model_id(self, model_pipeline: Any) -> Optional[str]:
        """(Internal Helper) Tries to extract the PyCaret model ID from a pipeline object."""
        if not model_pipeline:
            logger.debug("Cannot get model ID: model_pipeline object is None.")
            return None
        current_module = self.pycaret_module
        if not current_module:
             if hasattr(model_pipeline, 'predict_proba'): current_module = pyclf
             elif hasattr(model_pipeline, 'predict'): current_module = pyreg
             else:
                  logger.warning("Cannot get model ID: PyCaret module not set and cannot be inferred.")
                  return None
        try:
            estimator = None
            if hasattr(model_pipeline, 'named_steps') and 'actual_estimator' in model_pipeline.named_steps:
                 estimator = model_pipeline.named_steps['actual_estimator']
            elif hasattr(model_pipeline, 'steps'):
                 from sklearn.base import TransformerMixin
                 for _, step_obj in reversed(model_pipeline.steps):
                     if not isinstance(step_obj, TransformerMixin) and hasattr(step_obj, 'get_params'):
                          estimator = step_obj; break
                 if estimator is None: estimator = model_pipeline # Fallback
            elif hasattr(model_pipeline, 'get_params'):
                 estimator = model_pipeline
            else:
                 logger.warning(f"Could not extract estimator from pipeline type: {type(model_pipeline)}")
                 return None

            estimator_name = estimator.__class__.__name__

            try: # PyCaret internal mapping
                all_models_df = current_module.models()
                model_id_list = all_models_df[all_models_df['Class'] == estimator_name].index.tolist()
                if model_id_list:
                     found_id = model_id_list[0]
                     logger.debug(f"Mapped '{estimator_name}' to ID '{found_id}' via {current_module.__name__}.models()")
                     return found_id
            except Exception as models_lookup_error:
                 logger.debug(f"Could not look up model ID via {current_module.__name__}.models(): {models_lookup_error}")

            name_lower = estimator_name.lower()
            common_mappings = { # Fallback map
                'logisticregression': 'lr', 'ridgeclassifier': 'ridge', 'sgdclassifier': 'sgd',
                'kneighborsclassifier': 'knn', 'gaussiannb': 'gnb', 'decisiontreeclassifier': 'dt',
                'randomforestclassifier': 'rf', 'gradientboostingclassifier': 'gbr',
                'catboostclassifier': 'catboost', 'lgbmclassifier': 'lightgbm', 'xgbclassifier': 'xgboost',
                'extratreesclassifier': 'et', 'adaboostclassifier': 'ada', 'lineardiscriminantanalysis': 'lda',
                'quadraticdiscriminantanalysis': 'qda', 'svc': 'svm', 'nusvc': 'svm', 'mlpclassifier': 'mlp', 'dummyclassifier': 'dummy',
                'linearregression': 'lr', 'ridge': 'ridge', 'lasso': 'lasso', 'elasticnet': 'en', 'lars': 'lar',
                'lassolars': 'llars', 'orthogonalmatchingpursuit': 'omp', 'bayesianridge': 'br', 'ardregression': 'ard',
                'passiveaggressiveregressor': 'par', 'ransacregressor': 'ransac', 'theilsenregressor': 'tr', 'huberregressor': 'huber',
                'kneighborsregressor': 'knn', 'decisiontreeregressor': 'dt', 'randomforestregressor': 'rf',
                'extratreesregressor': 'et', 'adaboostregressor': 'ada', 'gradientboostingregressor': 'gbr',
                'catboostregressor': 'catboost', 'lgbmregressor': 'lightgbm', 'xgbregressor': 'xgboost',
                'svr': 'svm', 'nusvr': 'svm', 'dummyregressor': 'dummy', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost',
                'histgradientboostingregressor': 'gbr', 'histgradientboostingclassifier': 'gbr',
            }
            if name_lower in common_mappings:
                 found_id = common_mappings[name_lower]
                 logger.debug(f"Mapped '{estimator_name}' to ID '{found_id}' via fallback map.")
                 return found_id

            logger.warning(f"Could not reliably map estimator class '{estimator_name}' to a known PyCaret ID.")
            return None
        except Exception as e:
            logger.warning(f"Error occurred during model ID extraction: {e}", exc_info=True)
            return None


    # --- _generate_model_card ---
    def _generate_model_card(self, model_name, saved_model_base_path) -> str:
        """Generates a simple markdown model card string."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        active_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "N/A"
        reg_name = self.config.get('mlflow_registered_model_name', 'N/A')
        reg_stage = self.config.get('mlflow_model_stage', 'N/A')
        metric_name = self.optimize_metric or "N/A"
        metric_value = "N/A"
        if hasattr(self, 'latest_analysis_metrics') and self.latest_analysis_metrics:
             # Try to get the primary optimize metric, fallback to others
             metric_value_raw = self.latest_analysis_metrics.get(self.optimize_metric)
             if metric_value_raw is None and self.task_type == 'classification': metric_value_raw = self.latest_analysis_metrics.get('Accuracy')
             if metric_value_raw is None and self.task_type == 'regression': metric_value_raw = self.latest_analysis_metrics.get('R2')
             # Format value if found
             if isinstance(metric_value_raw, (int, float, np.number)):
                 metric_value = f"{metric_value_raw:.4f}"


        card = f"""
# Model Card: {model_name}

**Generated:** {timestamp}
**Session ID:** {self.session_id}
**MLflow Run ID:** {active_run_id}

## Model Details
- **Model Type:** `{model_name}` (Based on internal ID/stage)
- **Task Type:** {self.task_type or 'Unknown'}
- **PyCaret Version:** {pycaret.__version__}
- **Saved Path (Base):** `{saved_model_base_path}`
- **MLflow Registered Name:** {reg_name if self.config.get('register_model_in_mlflow') else 'Not Registered'}
- **MLflow Stage:** {reg_stage if self.config.get('register_model_in_mlflow') else 'N/A'}

## Training Data
- **Source Path:** `{self.config.get('data_file_path', 'N/A')}`
- **Target Column:** `{self.config.get('target_column', 'N/A')}`
- **Data Profile Report:** {'Logged as MLflow artifact' if self.training_data_profile_path and self.config.get("run_data_profiling") else 'Not Generated'}

## Performance Metrics (Hold-Out Set)
- **Primary Metric ({metric_name}):** {metric_value}
- *Note: Refer to logged metrics artifacts for detailed CV and hold-out results.*

## Configuration Snapshot (Key Params)
```json
{{
    "session_id": "{self.session_id}",
    "task_type": "{self.task_type}",
    "target_column": "{self.config.get('target_column')}",
    "sort_metric": "{self.sort_metric}",
    "optimize_metric": "{self.optimize_metric}",
    "baseline_folds": {self.config.get('baseline_folds')},
    "tuning_folds": {self.config.get('tuning_folds')},
    "tuning_iterations": {self.config.get('tuning_iterations')},
    "tuning_search_library": "{self.config.get('tuning_search_library')}",
    "tuning_search_algorithm": "{self.config.get('tuning_search_algorithm')}"
    # Add more key parameters as needed
}}
```
        """
        return card.strip()

    # --- _register_model ---
    def _register_model(self, model_uri: str, registered_model_name: str, description: str = "", stage: Optional[str] = None, tags: Optional[Dict] = None, await_registration_for=0) -> bool:
        """Registers a model in the MLflow Model Registry. Returns True on success."""
        if not mlflow.active_run():
            logger.error("Cannot register model - no active MLflow run.")
            return False
        if not registered_model_name:
            logger.error("Cannot register model - registered_model_name is empty.")
            return False

        logger.info(f"Attempting to register model in MLflow Model Registry...")
        logger.info(f" -> URI: {model_uri}")
        logger.info(f" -> Name: {registered_model_name}")
        logger.info(f" -> Stage: {stage or 'None'}")
        try:
            client = mlflow.tracking.MlflowClient()

            # Check if model already exists to avoid errors if name is reused across runs
            # Note: This creates the registered model name if it doesn't exist.
            try:
                 client.create_registered_model(registered_model_name)
                 logger.info(f"Created new registered model name: '{registered_model_name}'")
            except mlflow.exceptions.MlflowException as e:
                 if "RESOURCE_ALREADY_EXISTS" in str(e):
                      logger.info(f"Registered model name '{registered_model_name}' already exists.")
                 else: raise # Re-raise other MLflow exceptions

            # Register the model version
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
                await_registration_for=await_registration_for,
                tags=tags
            )
            logger.info(f"Model version registered: Name={model_version.name}, Version={model_version.version}")

            # Add description
            if description:
                client.update_model_version(
                    name=model_version.name,
                    version=model_version.version,
                    description=description
                )
                logger.info("Model version description updated.")

            # Transition stage if specified
            if stage and stage in ["Staging", "Production", "Archived"]:
                client.transition_model_version_stage(
                    name=model_version.name,
                    version=model_version.version,
                    stage=stage,
                    archive_existing_versions=(stage == "Production") # Archive others if moving to Prod
                )
                logger.info(f"Model version {model_version.version} transitioned to '{stage}'.")

            # Log registration info back to the run
            mlflow.log_param("mlflow_registered_model_name", model_version.name)
            mlflow.log_param("mlflow_registered_model_version", model_version.version)
            if stage: mlflow.log_param("mlflow_registered_model_stage", stage)

            return True

        except Exception as e:
            logger.error(f"Failed to register model '{registered_model_name}': {e}", exc_info=True)
            if mlflow.active_run():
                 mlflow.log_param("mlflow_registration_error", f"{type(e).__name__}: {str(e)[:500]}")
            return False

    # --- _check_data_drift ---
    def _check_data_drift(self, new_data: pd.DataFrame, report_save_name: Optional[str] = None) -> Optional[Dict]:
        """Checks for data drift between new data and the training data."""
        self.train_data = pd.read_csv(self.config['data_file_path']).drop(columns=[self.config['target_column']])
        if Report is None or DataDriftPreset is None:
            logger.error("Evidently library not installed. Cannot perform drift check.")
            return None
        if self.train_data is None:
            logger.error("Reference training data not available (was setup run/saved?). Cannot perform drift check.")
            return None

        logger.info("--- Checking for Data Drift ---")
        # Log drift check info to current run if active
        is_run_active = mlflow.active_run() is not None
        if is_run_active:
            mlflow.set_tag("drift_check_status", "started")

        try:
            reference_data = self.train_data.copy()
            current_data = new_data.copy()
            logger.info(f"Reference data shape: {reference_data.shape}, Current data shape: {current_data.shape}")

            # --- Align Columns (Essential for Evidently) ---
            ref_cols = list(reference_data.columns)
            cur_cols = list(current_data.columns)

            # Find common columns, maintaining order from reference data
            common_cols = [col for col in ref_cols if col in cur_cols]
            missing_in_current = list(set(ref_cols) - set(cur_cols))
            extra_in_current = list(set(cur_cols) - set(ref_cols))

            if missing_in_current or extra_in_current:
                logger.warning(f"Column mismatch for drift check. Using {len(common_cols)} common columns.")
                if missing_in_current: logger.warning(f" -> Columns missing in current data: {missing_in_current}")
                if extra_in_current: logger.warning(f" -> Columns extra in current data: {extra_in_current}")
                # Decide how to handle - Proceed with common columns?
                reference_data = reference_data[common_cols]
                current_data = current_data[common_cols]
                if not common_cols:
                    logger.error("No common columns found between reference and current data. Cannot perform drift check.")
                    return None

            # Ensure target column is handled (Evidently usually requires it to be same name or mapped)
            target_col = self.config.get("target_column")
            if target_col and target_col not in common_cols and target_col in ref_cols:
                 # Target is in reference but not current (typical prediction scenario)
                 logger.debug(f"Target column '{target_col}' present in reference, removed for drift check alignment.")
                 reference_data = reference_data.drop(columns=[target_col])
                 common_cols.remove(target_col) # Update common_cols list
            elif target_col and target_col in common_cols:
                # Target is in both - might need special handling if needed for drift metrics
                pass # Assume DataDriftPreset handles this correctly for now


            # --- Convert Categorical/Object Types (Often needed by Evidently metrics) ---
            # Evidently can sometimes struggle with pandas 'category' dtype directly
            for col in common_cols:
                if pd.api.types.is_categorical_dtype(reference_data[col].dtype):
                    logger.debug(f"Converting ref column '{col}' from category to object for drift.")
                    reference_data[col] = reference_data[col].astype(object)
                if pd.api.types.is_categorical_dtype(current_data[col].dtype):
                    logger.debug(f"Converting cur column '{col}' from category to object for drift.")
                    current_data[col] = current_data[col].astype(object)

            # --- Run Evidently Report ---
            logger.info(f"Running Evidently DataDriftPreset on {len(common_cols)} common columns...")
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report = drift_report.run(current_data=current_data, reference_data=reference_data) # Auto maps columns
            
            # drift_results = drift_report.dict()
            # drift_detected = drift_results['metrics'][0]['result']['dataset_drift']
            # num_drifted_features = drift_results['metrics'][0]['result']['number_of_drifted_columns']

            # logger.info(f"Data drift detected: {drift_detected}")
            # logger.info(f"Number of drifted features: {num_drifted_features}")
            # if is_run_active:
            #     mlflow.log_metric("data_drift_detected_flag", int(drift_detected))
            #     mlflow.log_metric("data_drift_num_drifted_features", num_drifted_features)
            #     mlflow.set_tag("drift_check_status", "completed")

            # --- Save and log report ---
            report_file = report_save_name or self.config.get("drift_report_name", "drift_report.html")
            full_report_path = os.path.join(self.report_save_dir, report_file) # Save in session reports dir
            try:
                 os.makedirs(self.report_save_dir, exist_ok=True)
                 drift_report.save_html(full_report_path)
                 logger.info(f"Drift report saved to {full_report_path}")
                 if is_run_active:
                     mlflow.log_artifact(full_report_path, artifact_path="reports")
                     mlflow.log_param("drift_report_path", full_report_path)
            except Exception as save_e:
                 logger.error(f"Failed to save drift report to {full_report_path}: {save_e}")

            # return {
            #     "drift_detected": drift_detected,
            #     "num_drifted_features": num_drifted_features,
            #     "report_path": full_report_path,
            #     "metrics_details": drift_results['metrics'] # Return detailed metrics if needed
            # }

        except Exception as e:
            logger.error(f"Data drift check failed: {e}", exc_info=True)
            if is_run_active:
                mlflow.set_tag("drift_check_status", "failed")
                mlflow.log_param("drift_check_error", f"{type(e).__name__}: {str(e)[:500]}")
            return None

    # ========================================================================
    # --- STEP 1: Setup, Compare Models, Analyze Baselines ---
    # ========================================================================
    def step1_setup_and_compare(self) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
        """
        Performs data loading, PROFILING (optional), setup, model comparison,
        baseline analysis, and saves the PyCaret experiment.

        Returns:
            Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
                - Path to the saved experiment file (.pkl), or None on failure.
                - DataFrame of comparison results (from pull()), can be None/empty.
                - Detected task type ('classification' or 'regression'), or None on failure.
        """
        step_status = "FAILED"
        active_mlflow_run_id = None
        experiment_save_path = None
        results_df_to_return = None
        final_task_type_return = None # Store the task type that ultimately worked
        step1_start_time = time.time()
        step_name = "Step1_SetupCompare"
        run_name = f"{step_name}_{self.session_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        initial_detected_task_type = None # Track the first detected type
        setup_attempt = 1
        max_setup_attempts = 2 # Allow initial attempt + 1 retry

        # --- Pre-checks ---
        required_keys = ["data_file_path", "target_column"]
        missing = [k for k in required_keys if k not in self.config]
        if missing:
             logger.error(f"{step_name} cannot start. Config missing keys: {missing}")
             return None, None, None

        try:
             # --- Ensure no orphaned runs ---
             self._ensure_no_active_mlflow_run(step_name)

             # Start MLflow run for this step
             with mlflow.start_run(run_name=run_name) as run:
                active_mlflow_run_id = run.info.run_id
                logger.info(f"Starting MLflow Run for {step_name}: {run_name} (ID: {active_mlflow_run_id})")
                # Log config subset
                mlflow.log_params({k: v for k, v in self.config.items() if isinstance(v, (str, int, float, bool)) and k not in ['session_id']})
                mlflow.log_dict(self.config.get("setup_params_extra", {}), "pycaret_setup_params_extra.json") # Log extra setup params
                mlflow.log_param("step_name", step_name)
                mlflow.log_param("session_output_dir", self.session_dir)
                mlflow.log_param("data_file_path", self.config.get("data_file_path"))
                mlflow.set_tags({"pycaret_version": pycaret.__version__, "session_id": str(self.session_id), f"{step_name}_status": "started"})

                # 1. Load Data
                logger.info("Step 1.1: Loading Data...")
                load_start = time.time()
                data_path = self.config["data_file_path"]
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data file not found: {data_path}")

                full_data = pd.read_csv(data_path)
                logger.info(f"Full data loaded from '{data_path}' shape: {full_data.shape}")

                # Apply dtype optimization if enabled
                if self.config.get('optimize_pandas_dtypes', True):
                     full_data = optimize_dtypes(full_data, self.config) # Use helper

                # --- Data Profiling ---
                if self.config.get("run_data_profiling") and ProfileReport:
                    profile_start = time.time()
                    logger.info("Step 1.1a: Generating Data Profile Report...")
                    try:
                        profile = ProfileReport(full_data, title="Data Profiling Report (Raw Data)", minimal=True)
                        report_file = self.config.get("profile_report_name", "data_profile_report.html")
                        self.training_data_profile_path = os.path.join(self.report_save_dir, report_file)
                        os.makedirs(self.report_save_dir, exist_ok=True)
                        profile.to_file(self.training_data_profile_path)
                        logger.info(f"Data profile report saved to {self.training_data_profile_path}")
                        mlflow.log_artifact(self.training_data_profile_path, artifact_path="reports")
                        mlflow.log_metric("data_profiling_duration_sec", time.time() - profile_start)
                    except Exception as e:
                        logger.error(f"Failed to generate data profile: {e}", exc_info=True)
                        mlflow.log_param("data_profiling_error", f"{type(e).__name__}: {str(e)[:500]}")
                        # Continue even if profiling fails


                # Apply sampling if configured
                self._manual_sampling_applied = False
                if self.config.get("use_sampling_in_setup", False):
                    logger.info("Applying manual sampling before PyCaret setup...")
                    sample_frac = self.config.get("sample_fraction")
                    sample_n = self.config.get("sample_n")
                    random_state = self.config.get("session_id")
                    
                    if len(full_data) >= 500000:
                        sample_frac = 0.01

                    if sample_n is not None and sample_n > 0:
                        sample_n = min(sample_n, len(full_data))
                        logger.info(f"Sampling {sample_n} rows (random_state={random_state}).")
                        self.data = full_data.sample(n=sample_n, random_state=random_state)
                        self._manual_sampling_applied = True
                    elif sample_frac is not None and 0 < sample_frac <= 1.0:
                        logger.info(f"Sampling fraction {sample_frac} (random_state={random_state}).")
                        self.data = full_data.sample(frac=sample_frac, random_state=random_state)
                        self._manual_sampling_applied = True
                    else:
                        logger.warning("Sampling enabled but no valid fraction/n. Using full data.")
                        self.data = full_data
                else:
                     logger.info("Using full dataset for PyCaret setup (sampling disabled).")
                     self.data = full_data

                logger.info(f"Data prepared for PyCaret setup shape: {self.data.shape}")
                self.data = self.data.reset_index(drop=True)
                mlflow.log_metric("load_data_duration_sec", time.time() - load_start)
                mlflow.log_param("pycaret_setup_input_rows", self.data.shape[0])


                # 2. Detect Task Type
                logger.info("Step 1.2: Initial Task Type Detection...")
                if not self._detect_task_type():
                     raise ValueError("Initial task type detection failed.")
                initial_detected_task_type = self.task_type
                logger.info(f"Initially detected task type: {initial_detected_task_type}")
                mlflow.log_param("initial_detected_task_type", initial_detected_task_type)

                # 3. Setup PyCaret (with retry logic)
                logger.info("Step 1.3: Setting up PyCaret...")
                setup_success = False
                last_setup_exception = None
                while setup_attempt <= max_setup_attempts and not setup_success:
                    logger.info(f"--- PyCaret Setup Attempt {setup_attempt}/{max_setup_attempts} with Task Type: {self.task_type} ---")
                    mlflow.set_tag("setup_attempt", setup_attempt) # Log attempt number
                    mlflow.log_param(f"setup_attempt_{setup_attempt}_task_type", self.task_type)

                    try:
                        setup_success = self._setup_pycaret() # This function uses self.task_type
                        if setup_success:
                            logger.info(f"PyCaret setup successful on attempt {setup_attempt} with task type: {self.task_type}")
                            final_task_type_return = self.task_type # Store the successful type
                            mlflow.log_param("final_confirmed_task_type", final_task_type_return) # Log final type
                        else:
                            # _setup_pycaret returned False, but didn't raise Exception (less common)
                            logger.warning(f"PyCaret setup attempt {setup_attempt} failed (returned False).")
                            # Force retry by breaking loop condition effectively
                            # setup_success remains False

                    except Exception as setup_exc:
                         logger.warning(f"PyCaret setup attempt {setup_attempt} with task type '{self.task_type}' raised an error: {setup_exc}", exc_info=True)
                         last_setup_exception = setup_exc
                         mlflow.log_param(f"setup_attempt_{setup_attempt}_error", f"{type(setup_exc).__name__}: {str(setup_exc)[:500]}")
                         # setup_success remains False

                    # --- Retry Logic ---
                    if not setup_success and setup_attempt < max_setup_attempts:
                        setup_attempt += 1
                        # Switch task type
                        original_type = self.task_type
                        alternate_type = 'regression' if original_type == 'classification' else 'classification'
                        logger.warning(f"Setup failed with {original_type}. Retrying with task type: {alternate_type}")

                        # Update runner state for the next attempt
                        self.task_type = alternate_type
                        self.config['task_type'] = self.task_type # Keep config consistent if needed later

                        # Explicitly check if metrics for alternate type exist BEFORE calling setup again
                        alt_sort_metric_key = f"sort_metric_{alternate_type}"
                        alt_opt_metric_key = f"optimize_metric_{alternate_type}"
                        if alt_sort_metric_key not in self.config or alt_opt_metric_key not in self.config:
                            err_msg = f"Cannot retry with task type '{alternate_type}'. Required config keys missing: {alt_sort_metric_key}, {alt_opt_metric_key}"
                            logger.error(err_msg)
                            mlflow.log_param(f"setup_attempt_{setup_attempt}_error", "Missing config for alternate task type metrics")
                            raise ValueError(err_msg) from last_setup_exception # Fail the step now

                        # Reset internal state variables affected by _setup_pycaret
                        self.pycaret_module = None
                        self.sort_metric = None
                        self.optimize_metric = None
                        self.setup_env = None
                        self.preprocessor = None
                        self.train_data = None
                        self.test_data = None

                    elif not setup_success and setup_attempt >= max_setup_attempts:
                        logger.error(f"PyCaret setup failed after {max_setup_attempts} attempts.")
                        # Re-raise the last exception encountered or a generic one if _setup returned False
                        if last_setup_exception:
                            raise RuntimeError("PyCaret setup failed after multiple attempts.") from last_setup_exception
                        else:
                            raise RuntimeError("PyCaret setup failed after multiple attempts (returned False).")
                    else: # setup_success is True
                        break # Exit the while loop

                # If loop finishes without setup_success (shouldn't happen due to raise)
                if not setup_success:
                     raise RuntimeError("PyCaret setup definitively failed.")

                # --- Steps 4, 5, 6 (Proceed only if setup succeeded) ---
                
                # 4. Compare Models
                logger.info("Step 1.4: Comparing Models...")
                if not self._compare_models(): # Handles own logging
                     logger.warning("Model comparison had issues or failed.")
                results_df_to_return = self.results_df.copy() if self.results_df is not None else None

                # 5. Analyze Baseline Models (Optional)
                logger.info("Step 1.5: Analyzing Baseline Models...")
                # _analyze_baseline_models logs to parent run and handles nested runs
                if not self._analyze_baseline_models():
                    # Baseline failure might not be critical? Log and continue.
                    logger.warning("Baseline model analysis encountered errors.")

                # 6. Save Experiment
                logger.info("Step 1.6: Saving PyCaret Experiment...")
                exp_save_start = time.time()
                experiment_filename = "pycaret_experiment.pkl"
                experiment_save_path = os.path.join(self.experiment_save_dir, experiment_filename)
                if self.pycaret_module and self.setup_env: # Check setup_env exists too
                     # Use save_experiment attached to the setup environment instance
                     self.setup_env.save_experiment(path_or_file=experiment_save_path)
                     if not os.path.exists(experiment_save_path):
                          raise RuntimeError("save_experiment did not create the file.")
                     logger.info(f"Experiment saved successfully: {experiment_save_path}")
                     mlflow.log_artifact(experiment_save_path, artifact_path="experiment")
                     mlflow.log_param("step1_experiment_save_path", experiment_save_path)
                     mlflow.log_metric("save_experiment_duration_sec", time.time() - exp_save_start)
                else:
                     raise RuntimeError("PyCaret module or setup env not ready, cannot save.")

                step_status = "COMPLETED"
                logger.info(f"--- {step_name} COMPLETED ---")

        except Exception as e:
             step_status = "FAILED"
             logger.error(f"--- {step_name} FAILED: {e} ---", exc_info=True)
             if active_mlflow_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == active_mlflow_run_id:
                 try: mlflow.set_tag("error_message", f"{type(e).__name__}: {str(e)[:500]}")
                 except Exception as log_err: logger.error(f"Failed log error tag to MLflow: {log_err}")
             experiment_save_path = None
             results_df_to_return = None
             final_task_type_return = None # Ensure None on failure

        finally:
            step1_total_duration = time.time() - step1_start_time
            logger.info(f"{step_name} finished. Status: {step_status}. Duration: {step1_total_duration:.2f}s.")
            if active_mlflow_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == active_mlflow_run_id:
                try:
                    mlflow.log_metric("step1_total_duration_sec", step1_total_duration)
                    mlflow.set_tag(f"{step_name}_status", step_status)
                    # Log final task type even on failure if it was determined before error
                    if final_task_type_return:
                         mlflow.log_param("final_confirmed_task_type", final_task_type_return)
                    elif self.task_type and setup_attempt > 1: # Log the last attempted type if failed on retry
                         mlflow.log_param("last_attempted_task_type_on_fail", self.task_type)
                    elif initial_detected_task_type: # Log initial if failed before retry
                        mlflow.log_param("initial_detected_task_type_on_fail", initial_detected_task_type)

                    mlflow.end_run()
                    logger.info(f"Ended MLflow Run {active_mlflow_run_id} ({step_name}).")
                except Exception as final_log_err:
                    logger.error(f"Failed log final status/duration or end MLflow run {active_mlflow_run_id}: {final_log_err}")
            elif mlflow.active_run():
                current_run_id = mlflow.active_run().info.run_id
                logger.warning(f"Unexpected MLflow run ({current_run_id}) active at end of {step_name}. Ending it.")
                try: mlflow.end_run(status="KILLED")
                except Exception as end_err: logger.error(f"Failed end unexpected run {current_run_id}: {end_err}")

        # Return the task type that WORKED
        return experiment_save_path, results_df_to_return, final_task_type_return


    # ========================================================================
    # --- STEP 2: Tune User-Specified Model and Analyze ---
    # ========================================================================
    def step2_tune_and_analyze_model(self, experiment_path: str, model_id_to_tune: str) -> Optional[str]:
        """
        Loads experiment, tunes SPECIFIED model, analyzes it (plots, holdout metrics),
        and saves the tuned model artifact.

        Args:
            experiment_path: Path to the saved PyCaret experiment (.pkl) from Step 1.
            model_id_to_tune: String ID of the model to tune (e.g., 'rf', 'lightgbm').

        Returns:
            Optional[str]: Base path (without .pkl) of the saved TUNED model artifact, or None.
        """
        step_status = "FAILED"
        active_mlflow_run_id = None
        tuned_model_save_path_base = None
        step2_start_time = time.time()
        step_name = f"Step2_Tune_{model_id_to_tune}"
        run_name = f"{step_name}_{self.session_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.tuned_best_model = None
        self.last_tuned_cv_results_df = None
        self.last_tuned_best_params = None
        self.last_feature_importance_plot_path = None
        
        # --- Pre-checks ---
        if not experiment_path or not os.path.exists(experiment_path):
            logger.error(f"{step_name} cannot start. Experiment path '{experiment_path}' invalid/missing.")
            return None
        if not model_id_to_tune or not isinstance(model_id_to_tune, str):
            logger.error(f"{step_name} cannot start. model_id_to_tune must be non-empty string.")
            return None
        # Task type should be passed via config by the orchestrator based on Step 1's output
        if "task_type" not in self.config or not self.config["task_type"]:
             logger.warning("task_type not found in config for Step 2. Fallback loading might fail.")

        try:
            # --- Ensure no orphaned runs ---
            self._ensure_no_active_mlflow_run(step_name)

            # Start MLflow run for this step
            with mlflow.start_run(run_name=run_name) as run:
                active_mlflow_run_id = run.info.run_id
                logger.info(f"Starting MLflow Run for {step_name}: {run_name} (ID: {active_mlflow_run_id})")
                mlflow.log_param("step_name", step_name)
                mlflow.log_param("session_id", str(self.session_id))
                mlflow.log_param("input_experiment_path", experiment_path)
                mlflow.log_param("model_id_to_tune", model_id_to_tune)
                mlflow.set_tags({"session_id": str(self.session_id), "pycaret_version": pycaret.__version__, f"{step_name}_status": "started"})

                # 1. Load Experiment
                logger.info("Step 2.1: Loading Experiment...")
                load_exp_start = time.time()
                loaded_env = None
                task_type_for_load = self.config.get("task_type")
                pycaret_module_for_load = None

                if task_type_for_load == 'classification': pycaret_module_for_load = pyclf
                elif task_type_for_load == 'regression': pycaret_module_for_load = pyreg

                if pycaret_module_for_load:
                    logger.info(f"Attempting load with pre-defined task type: {task_type_for_load}")
                    try: # Pass data to load_experiment for consistency
                        data_for_load = pd.read_csv(self.config['data_file_path'])
                        if len(data_for_load) >= 500000:
                            random_state = self.config.get("session_id")
                            data_for_load = data_for_load.sample(frac=0.01, random_state=random_state)
                        
                        data_for_load = data_for_load.reset_index(drop=True)
                        if self.config.get('optimize_pandas_dtypes', True):
                            data_for_load = optimize_dtypes(data_for_load, self.config) # optimize dtypes
                        loaded_env = pycaret_module_for_load.load_experiment(experiment_path, data=data_for_load)
                        self.pycaret_module = pycaret_module_for_load
                        self.task_type = task_type_for_load
                    except Exception as load_err:
                         raise RuntimeError(f"Failed load experiment with type {task_type_for_load}") from load_err
                else: # Fallback loading
                    logger.warning("Task type not explicit. Attempting fallback load...")
                    try:
                         data_for_load = pd.read_csv(self.config['data_file_path'])
                         if len(data_for_load) >= 500000:
                            random_state = self.config.get("session_id")
                            data_for_load = data_for_load.sample(frac=0.01, random_state=random_state)
                        
                         data_for_load = data_for_load.reset_index(drop=True)
                         if self.config.get('optimize_pandas_dtypes', True): 
                             data_for_load = optimize_dtypes(data_for_load, self.config)
                         loaded_env = pyclf.load_experiment(experiment_path, data=data_for_load)
                         self.pycaret_module = pyclf; self.task_type = 'classification'
                    except Exception:
                         logger.warning("Failed clf load, trying reg...")
                         try:
                             data_for_load = pd.read_csv(self.config['data_file_path'])
                             
                             if len(data_for_load) >= 500000:
                                random_state = self.config.get("session_id")
                                data_for_load = data_for_load.sample(frac=0.01, random_state=random_state)

                             data_for_load = data_for_load.reset_index(drop=True)
                             if self.config.get('optimize_pandas_dtypes', True): 
                                 data_for_load = optimize_dtypes(data_for_load, self.config)
                             loaded_env = pyreg.load_experiment(experiment_path, data=data_for_load)
                             self.pycaret_module = pyreg; self.task_type = 'regression'
                         except Exception as load_err_reg:
                             raise RuntimeError(f"Could not load experiment from {experiment_path}") from load_err_reg

                # --- Update self.setup_env AFTER loading ---
                self.setup_env = loaded_env
                logger.info(f"Experiment loaded. Task Type: {self.task_type}")
                mlflow.log_param("loaded_task_type", self.task_type)
                mlflow.log_metric("load_experiment_duration_sec", time.time() - load_exp_start)

                # Re-set optimize metric based on loaded/determined task type
                metric_key_base = f"optimize_metric_{self.task_type}"
                if metric_key_base not in self.config:
                     raise ValueError(f"Optimize metric '{metric_key_base}' not in config for task {self.task_type}.")
                self.optimize_metric = self.config[metric_key_base]
                mlflow.log_param("tune_optimize_metric", self.optimize_metric)
                

                # 2. Tune Specified Model
                logger.info(f"Step 2.2: Tuning model '{model_id_to_tune}' optimizing '{self.optimize_metric}'...")
                # Create base model instance first (as per Code 1 fix)
                create_start = time.time()
                try:
                    logger.info(f"Creating base model instance for '{model_id_to_tune}'...")
                    create_fold_count = self.config.get("tuning_folds", 5) # Use tuning folds
                    base_model_to_tune = self.pycaret_module.create_model(
                        model_id_to_tune, fold=create_fold_count, verbose=False
                    )
                    if base_model_to_tune is None:
                        raise RuntimeError(f"create_model returned None for '{model_id_to_tune}'.")
                    create_duration = time.time() - create_start
                    logger.info(f"Base model created in {create_duration:.2f}s.")
                    mlflow.log_metric("create_base_model_duration_sec", create_duration)
                except Exception as create_err:
                    raise RuntimeError(f"Failed create base model '{model_id_to_tune}' for tuning") from create_err

                # Prepare tuning parameters
                tune_start = time.time()
                tune_params = {
                    "optimize": self.optimize_metric,
                    "fold": self.config.get("tuning_folds", 5),
                    "n_iter": self.config.get("tuning_iterations", 10),
                    "search_library": self.config.get("tuning_search_library", 'scikit-learn'),
                    "search_algorithm": self.config.get("tuning_search_algorithm", 'random'),
                    "verbose": self.config.get("pycaret_tune_verbose", False),
                    **({"custom_grid": self.config["custom_grid"]} if "custom_grid" in self.config else {})
                }
                if tune_params["search_algorithm"].lower() not in ['random', 'bayesian', 'optuna', 'skopt']:
                    tune_params.pop("n_iter", None)

                # Execute Tuning
                try:
                    logger.info(f"Tuning the created '{model_id_to_tune}' model instance...")
                    tuned_model = self.pycaret_module.tune_model(
                        estimator=base_model_to_tune, # Pass created model object
                        **tune_params
                    )
                    if tuned_model is None:
                        raise RuntimeError(f"tune_model failed for '{model_id_to_tune}'.")
                    self.tuned_best_model = tuned_model # Store temporarily

                    tuning_duration = time.time() - tune_start
                    logger.info(f"Model tuning completed in {tuning_duration:.2f}s.")

                    self.last_tuned_cv_results_df = self.pycaret_module.pull()
                    tuned_cv_results_df = self.last_tuned_cv_results_df # Assign to local var for return

                    # Attempt to extract best parameters
                    try:
                        final_estimator_step = None
                        if hasattr(self.tuned_best_model, 'steps'):
                            final_estimator_step = self.tuned_best_model.steps[-1][1]
                        else:
                            final_estimator_step = self.tuned_best_model

                        if final_estimator_step and hasattr(final_estimator_step, 'get_params'):
                            # Get params specifically from the estimator, not the whole pipeline
                            self.last_tuned_best_params = final_estimator_step.get_params(deep=False)
                            best_params_dict = self.last_tuned_best_params # Assign for return
                            logger.info(f"Extracted best params: {best_params_dict}")
                            if best_params_dict: mlflow.log_dict(best_params_dict, "tuned_best_params.json")
                        else:
                            logger.warning("Could not extract final estimator or params from tuned model.")

                    except Exception as param_ex:
                        logger.warning(f"Error extracting best params: {param_ex}")
                    
                    # Log tuning params and CV results
                    mlflow.log_params({f"tune_{k}":v for k,v in tune_params.items() if k != 'custom_grid'}) # Log simple params
                    if "custom_grid" in tune_params: mlflow.log_dict(tune_params["custom_grid"], "tuning_custom_grid.json")
                    mlflow.log_metric("tune_duration_seconds", tuning_duration)
                    tuned_results = self.pycaret_module.pull()
                    if tuned_results is not None and not tuned_results.empty:
                        if 'Mean' in tuned_results.index and self.optimize_metric in tuned_results.columns:
                            tuned_metric_val = tuned_results.loc['Mean', self.optimize_metric]
                            if isinstance(tuned_metric_val, (int, float, np.number)):
                                mlflow.log_metric(f"tuned_cv_mean_{self.optimize_metric}", float(tuned_metric_val))
                        try: # Log results table
                            tuned_results_html = tuned_results.to_html(escape=False, max_rows=20)
                            mlflow.log_text(tuned_results_html, f"tuned_{model_id_to_tune}_cv_results.html")
                        except Exception as log_err: logger.error(f"Failed log tune_model results table: {log_err}")
                    else: logger.warning("tune_model did not return results from pull().")

                except Exception as e:
                    logger.error(f"Failed during model tuning for {model_id_to_tune}: {e}", exc_info=True)
                    raise # Propagate error


                # 3. Analyze Tuned Model (Optional)
                if self.config.get("analyze_tuned_step2", True):
                    logger.info("Step 2.3: Analyzing Tuned Model...")
                    # _analyze_tuned_model logs duration/plots/metrics internally
                    if not self._analyze_tuned_model():
                        logger.warning("Analysis of tuned model had non-fatal errors.")
                        
                    feature_importance_plot_path = self.last_feature_importance_plot_path
                else:
                    logger.info("Skipping tuned model analysis in Step 2 (config).")


                # 4. Save Tuned Model Artifact
                logger.info("Step 2.4: Saving TUNED model artifact...")
                save_tuned_start = time.time()
                # Use the model ID that was actually tuned for saving/logging
                tuned_model_save_path_base = self._save_model_artifact(
                    model_object=self.tuned_best_model,
                    model_stage="tuned",
                    model_id=model_id_to_tune # Use the ID provided to the step
                )
                if not tuned_model_save_path_base:
                     raise RuntimeError("Failed to save the tuned model artifact.")

                mlflow.log_param("step2_tuned_model_save_path_base", tuned_model_save_path_base)
                mlflow.log_metric("save_tuned_model_duration_sec", time.time() - save_tuned_start)

                step_status = "COMPLETED"
                logger.info(f"--- {step_name} COMPLETED ---")

        except Exception as e:
             step_status = "FAILED"
             logger.error(f"--- {step_name} FAILED: {e} ---", exc_info=True)
             if active_mlflow_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == active_mlflow_run_id:
                 try: mlflow.set_tag("error_message", f"{type(e).__name__}: {str(e)[:500]}")
                 except Exception as log_err: logger.error(f"Failed log error tag to MLflow: {log_err}")
             tuned_model_save_path_base = None

        finally:
            step2_total_duration = time.time() - step2_start_time
            logger.info(f"{step_name} finished. Status: {step_status}. Duration: {step2_total_duration:.2f}s.")
            if active_mlflow_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == active_mlflow_run_id:
                try:
                    mlflow.log_metric("step2_total_duration_sec", step2_total_duration)
                    mlflow.set_tag(f"{step_name}_status", step_status)
                    mlflow.end_run() # End run
                    logger.info(f"Ended MLflow Run {active_mlflow_run_id} ({step_name}).")
                except Exception as final_log_err:
                    logger.error(f"Failed log final status/duration or end MLflow run {active_mlflow_run_id}: {final_log_err}")
            elif mlflow.active_run(): # Safety check
                 current_run_id = mlflow.active_run().info.run_id
                 logger.warning(f"Unexpected MLflow run ({current_run_id}) active at end of {step_name}. Ending it.")
                 try: mlflow.end_run(status="KILLED")
                 except Exception as end_err: logger.error(f"Failed end unexpected run {current_run_id}: {end_err}")

        return tuned_model_save_path_base, best_params_dict, tuned_cv_results_df, feature_importance_plot_path


    # ========================================================================
    # --- STEP 3: Finalize Model, Save, Card, Register ---
    # ========================================================================
    def step3_finalize_and_save_model(self, experiment_path: str, tuned_model_path_base: str) -> Optional[str]:
        """
        Loads experiment, loads TUNED model, finalizes it, saves FINAL artifact,
        generates model card (optional), and registers in MLflow (optional).

        Args:
            experiment_path: Path to saved PyCaret experiment (.pkl) from Step 1.
            tuned_model_path_base: Base path (without .pkl) of saved TUNED model artifact from Step 2.

        Returns:
            Optional[str]: Base path (without .pkl) of the saved FINAL model artifact, or None.
        """
        step_status = "FAILED"
        active_mlflow_run_id = None
        final_model_save_path_base = None
        step3_start_time = time.time()

        # Extract model ID from tuned path for naming run/logging
        tuned_model_name_part = "unknown"
        final_model_id_for_naming = "unknown"
        if tuned_model_path_base:
             try:
                 fname = os.path.basename(tuned_model_path_base)
                 parts = fname.split('_')
                 if len(parts) > 1 and parts[0] == "tuned":
                     final_model_id_for_naming = parts[1] # Get the ID part
                     tuned_model_name_part = final_model_id_for_naming
                 else: # Fallback parsing
                      tuned_model_name_part = parts[1] if len(parts) > 1 else fname
                      final_model_id_for_naming = tuned_model_name_part

             except Exception: logger.warning("Could not parse model ID from tuned path.")

        step_name = f"Step3_Finalize_{tuned_model_name_part}"
        run_name = f"{step_name}_{self.session_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # --- Pre-checks ---
        if not experiment_path or not os.path.exists(experiment_path):
            logger.error(f"{step_name} cannot start. Experiment path '{experiment_path}' invalid/missing.")
            return None
        if not tuned_model_path_base:
             logger.error(f"{step_name} cannot start. Tuned model path base empty.")
             return None
        tuned_model_pkl_path = f"{tuned_model_path_base}.pkl"
        if not os.path.exists(tuned_model_pkl_path):
             logger.error(f"{step_name} cannot start. Tuned model PKL not found: {tuned_model_pkl_path}")
             return None
        # Task type needed for loading
        if "task_type" not in self.config or not self.config["task_type"]:
             logger.warning("task_type not in config for Step 3. Fallback loading might fail.")


        try:
            # --- Ensure no orphaned runs ---
            self._ensure_no_active_mlflow_run(step_name)

            # Start MLflow run for this step
            with mlflow.start_run(run_name=run_name) as run:
                active_mlflow_run_id = run.info.run_id
                logger.info(f"Starting MLflow Run for {step_name}: {run_name} (ID: {active_mlflow_run_id})")
                mlflow.log_param("step_name", step_name)
                mlflow.log_param("session_id", str(self.session_id))
                mlflow.log_param("input_experiment_path", experiment_path)
                mlflow.log_param("input_tuned_model_path_base", tuned_model_path_base)
                mlflow.log_param("finalized_model_base_id", final_model_id_for_naming)
                mlflow.set_tags({"session_id": str(self.session_id), "pycaret_version": pycaret.__version__, f"{step_name}_status": "started"})


                # 1. Load Experiment (needed for finalize_model context)
                logger.info("Step 3.1: Loading Experiment...")
                load_exp_start = time.time()
                # --- Determine task type for loading (Same logic as Step 2) ---
                task_type_for_load = self.config.get("task_type")
                pycaret_module_for_load = None
                if task_type_for_load == 'classification': pycaret_module_for_load = pyclf
                elif task_type_for_load == 'regression': pycaret_module_for_load = pyreg

                if pycaret_module_for_load:
                    try:
                        data_for_load = pd.read_csv(self.config['data_file_path'])
                        if self.config.get('optimize_pandas_dtypes', True): 
                            data_for_load = optimize_dtypes(data_for_load, self.config)
                        loaded_env = pycaret_module_for_load.load_experiment(experiment_path, data=data_for_load)
                    except Exception as le: raise RuntimeError(f"Failed load with type {task_type_for_load}") from le
                else: # Fallback
                    logger.warning("Task type not explicit. Attempting fallback load...")
                    try:
                        data_for_load = pd.read_csv(self.config['data_file_path'])
                        if self.config.get('optimize_pandas_dtypes', True): 
                            data_for_load = optimize_dtypes(data_for_load, self.config)
                        loaded_env = pyclf.load_experiment(experiment_path, data=data_for_load); pycaret_module_for_load=pyclf; task_type_for_load='classification'
                    except Exception:
                         logger.warning("Failed clf load, trying reg...")
                         try:
                             data_for_load = pd.read_csv(self.config['data_file_path'])
                             if self.config.get('optimize_pandas_dtypes', True): 
                                 data_for_load = optimize_dtypes(data_for_load, self.config)
                             loaded_env = pyreg.load_experiment(experiment_path, data=data_for_load); pycaret_module_for_load=pyreg; task_type_for_load='regression'
                         except Exception as ler: raise RuntimeError(f"Could not load experiment from {experiment_path}") from ler

                # --- Update self state AFTER loading ---
                self.pycaret_module = pycaret_module_for_load
                self.task_type = task_type_for_load
                self.setup_env = loaded_env # Ensure setup_env is set for finalize context
                logger.info(f"Experiment loaded. Task Type: {self.task_type}")
                mlflow.log_param("loaded_task_type", self.task_type)
                mlflow.log_metric("load_experiment_duration_sec", time.time() - load_exp_start)


                # 2. Load Tuned Model Pipeline
                logger.info("Step 3.2: Loading TUNED model pipeline...")
                load_tuned_start = time.time()
                # Use the module determined above and the _load_model helper for consistency
                tuned_model_pipeline = self.pycaret_module.load_model(tuned_model_path_base, verbose=False)
                if tuned_model_pipeline is None:
                     raise RuntimeError(f"load_model returned None for tuned model: {tuned_model_path_base}")
                logger.info("Tuned model loaded successfully.")
                mlflow.log_metric("load_tuned_model_duration_sec", time.time() - load_tuned_start)

                # 3. Finalize Model
                logger.info("Step 3.3: Finalizing model...")
                finalize_start = time.time()
                # finalize_model uses the context (data) from the currently loaded experiment (self.setup_env)
                final_model_obj = self.pycaret_module.finalize_model(tuned_model_pipeline)
                if final_model_obj is None:
                     raise RuntimeError("finalize_model failed or returned None.")
                self.final_model = final_model_obj # Store temporarily

                finalizing_duration = time.time() - finalize_start
                logger.info(f"Model finalized in {finalizing_duration:.2f} seconds.")
                mlflow.log_metric("finalize_duration_seconds", finalizing_duration)


                # 4. Save Final Model Artifact
                logger.info("Step 3.4: Saving FINAL model artifact...")
                save_final_start = time.time()
                # Use the determined model ID for consistent naming
                final_model_save_path_base = self._save_model_artifact(
                    model_object=self.final_model,
                    model_stage="final",
                    model_id=final_model_id_for_naming # Use parsed ID
                )
                if not final_model_save_path_base:
                     raise RuntimeError("Failed to save the final model artifact.")
                mlflow.log_param("step3_final_model_save_path_base", final_model_save_path_base)
                mlflow.log_metric("save_final_model_duration_sec", time.time() - save_final_start)


                # --- Generate Model Card ---
                if self.config.get("generate_model_card", False):
                    logger.info("Step 3.5: Generating Model Card...")
                    try:
                        # Model card needs metrics - ideally passed from step 2 analysis or re-evaluated here
                        # Using self.latest_analysis_metrics stored in _analyze_tuned_model
                        card_content = self._generate_model_card(final_model_id_for_naming, final_model_save_path_base)
                        card_path = f"{final_model_save_path_base}_model_card.md"
                        with open(card_path, "w", encoding='utf-8') as f:
                            f.write(card_content)
                        mlflow.log_artifact(card_path, artifact_path="reports") # Log card to reports
                        logger.info(f"Model card saved and logged: {card_path}")
                        mlflow.log_param("model_card_generated", True)
                    except Exception as card_e:
                         logger.error(f"Failed to generate or log model card: {card_e}")
                         mlflow.log_param("model_card_error", f"{type(card_e).__name__}")


                # --- Register Model in MLflow ---
                if self.config.get("register_model_in_mlflow", False):
                    logger.info("Step 3.6: Registering Model in MLflow Registry...")
                    # Need the artifact URI. Construct it based on how _save_model_artifact logs.
                    final_model_pkl_filename = os.path.basename(f"{final_model_save_path_base}.pkl")
                    model_artifact_uri = f"runs:/{active_mlflow_run_id}/models/final/{final_model_pkl_filename}"

                    reg_success = self._register_model(
                        model_uri=model_artifact_uri, # Use the URI of the logged PKL artifact
                        registered_model_name=self.config.get("mlflow_registered_model_name", f"automl_{final_model_id_for_naming}"),
                        description=f"Finalized AutoML model from session {self.session_id}. Model ID: {final_model_id_for_naming}. Run: {active_mlflow_run_id}",
                        stage=self.config.get("mlflow_model_stage", "Staging"),
                        tags={"automl_session_id": str(self.session_id), "model_id": final_model_id_for_naming, "step": "step3"}
                    )
                    if reg_success:
                        logger.info("Model registration successful.")
                    else:
                         logger.warning("Model registration in MLflow failed.")


                step_status = "COMPLETED"
                logger.info(f"--- {step_name} COMPLETED ---")

        except Exception as e:
             step_status = "FAILED"
             logger.error(f"--- {step_name} FAILED: {e} ---", exc_info=True)
             if active_mlflow_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == active_mlflow_run_id:
                 try: mlflow.set_tag("error_message", f"{type(e).__name__}: {str(e)[:500]}")
                 except Exception as log_err: logger.error(f"Failed log error tag to MLflow: {log_err}")
             final_model_save_path_base = None # Ensure None on failure

        finally:
            step3_total_duration = time.time() - step3_start_time
            logger.info(f"{step_name} finished. Status: {step_status}. Duration: {step3_total_duration:.2f}s.")
            if active_mlflow_run_id and mlflow.active_run() and mlflow.active_run().info.run_id == active_mlflow_run_id:
                try:
                    mlflow.log_metric("step3_total_duration_sec", step3_total_duration)
                    mlflow.set_tag(f"{step_name}_status", step_status)
                    mlflow.end_run() # End run
                    logger.info(f"Ended MLflow Run {active_mlflow_run_id} ({step_name}).")
                except Exception as final_log_err:
                    logger.error(f"Failed log final status/duration or end MLflow run {active_mlflow_run_id}: {final_log_err}")
            elif mlflow.active_run(): # Safety check
                 current_run_id = mlflow.active_run().info.run_id
                 logger.warning(f"Unexpected MLflow run ({current_run_id}) active at end of {step_name}. Ending it.")
                 try: mlflow.end_run(status="KILLED")
                 except Exception as end_err: logger.error(f"Failed end unexpected run {current_run_id}: {end_err}")

        return final_model_save_path_base # Return absolute base path


    # ========================================================================
    # --- PREDICTION (Integrate Drift Check) ---
    # ========================================================================
    def predict_on_new_data(self, new_data: pd.DataFrame, model_base_path: str) -> Optional[pd.DataFrame]:
        """
        Makes predictions using a saved FINAL model pipeline.
        Includes optional data drift check before prediction.

        Args:
            new_data: DataFrame with features matching training data.
            model_base_path: Absolute base path (without .pkl) to saved FINAL model.

        Returns:
            DataFrame with predictions appended, or None if prediction fails.
        """
        logger.info(f"--- Making Predictions using model: {model_base_path} ---")
        pred_start_time = time.time()
        step_name = "Predict"

        # --- Pre-checks ---
        if new_data is None or new_data.empty:
             logger.error("Prediction failed: input data is None or empty.")
             return None
        if not isinstance(new_data, pd.DataFrame):
             logger.error(f"Prediction failed: input data must be pandas DataFrame, got {type(new_data)}.")
             return None
        if not model_base_path:
             logger.error("Prediction failed: model_base_path cannot be empty.")
             return None
        # Use helper to load model and get task type from metadata
        loaded_pipeline, loaded_task_type = self._load_model_and_metadata(model_base_path)

        if loaded_pipeline is None:
             logger.error("Prediction failed: Could not load model pipeline.")
             return None
        if loaded_task_type is None:
             # This should ideally not happen if metadata exists, but handle defensively
             logger.error("Prediction failed: Could not determine task type from model metadata.")
             return None

        # Determine the correct PyCaret module based on loaded type
        pred_module = pyclf if loaded_task_type == 'classification' else pyreg
        logger.info(f"Using prediction module: {pred_module.__name__}")

        # --- Data Drift Check ---
        if self.config.get("enable_drift_check_on_predict", True):
             logger.info("Performing data drift check before prediction...")
             # Ensure self.train_data is available (should be set in step 1)
             drift_results = self._check_data_drift(new_data=new_data.copy()) # Pass copy
             if drift_results:
                 logger.info(f"Drift Check Results: Detected={drift_results.get('drift_detected')}, DriftedFeatures={drift_results.get('num_drifted_features')}")
                 if drift_results.get("drift_detected"):
                     logger.warning("<<< POTENTIAL DATA DRIFT DETECTED! Prediction results may be unreliable. >>>")
                     if self.config.get("halt_prediction_on_drift", False): # Add config option
                         logger.error("Halting prediction due to detected data drift.")
                         return None
             else:
                 logger.warning("Data drift check failed or could not be performed.")

        # --- Load model and Predict ---
        try:
            logger.info("Preparing prediction data...")
            predict_data = new_data.copy()
            # Target column should ideally NOT be in prediction data, but check/remove just in case
            target_col = self.config.get("target_column")
            if target_col and target_col in predict_data.columns:
                 logger.warning(f"Target column '{target_col}' found in prediction input. Removing it.")
                 predict_data = predict_data.drop(columns=[target_col])

            # Optimize dtypes if enabled
            if self.config.get('optimize_pandas_dtypes', True):
                 logger.info("Optimizing dtypes of new data before prediction...")
                 predict_data = optimize_dtypes(predict_data, self.config)

            # Make predictions
            logger.info(f"Generating predictions on data shape: {predict_data.shape}")
            # predict_model applies the transformations within the loaded pipeline
            predictions_out = pred_module.predict_model(loaded_pipeline, data=predict_data, verbose=False)
            logger.info(f"Predictions generated. Output shape: {predictions_out.shape}")

            # --- Rename prediction columns based on config ---
            pred_label_col_default = 'prediction_label'
            pred_score_col_default = 'prediction_score'
            target_label_col = self.config.get("prediction_target_column_name", pred_label_col_default)
            target_score_col = self.config.get("prediction_score_column_name", pred_score_col_default)

            rename_map = {}
            if loaded_task_type == 'classification':
                if pred_label_col_default in predictions_out.columns: rename_map[pred_label_col_default] = target_label_col
                if pred_score_col_default in predictions_out.columns: rename_map[pred_score_col_default] = target_score_col
            elif loaded_task_type == 'regression':
                 if pred_label_col_default in predictions_out.columns: rename_map[pred_label_col_default] = target_label_col

            if rename_map:
                 predictions_out = predictions_out.rename(columns=rename_map)
                 logger.info(f"Renamed prediction output columns: {rename_map}")

            logger.info(f"Prediction step duration: {time.time() - pred_start_time:.2f} seconds.")
            return predictions_out

        except ValueError as ve:
             logger.error(f"Prediction failed: ValueError likely data schema mismatch. Error: {ve}", exc_info=True)
             logger.error(f"Prediction data columns: {predict_data.columns.tolist()}")
             return None
        except Exception as e:
             logger.error(f"Prediction failed: Unexpected error: {e}", exc_info=True)
             logger.error(f"Prediction failed on data shape: {new_data.shape}")
             return None


    # ========================================================================
    # --- SHAP Explanation Method ---
    # ========================================================================
    def explain_prediction(self, model_base_path: str, data_instance: pd.DataFrame) -> Optional[Dict]:
        """
        Explains a single prediction using SHAP. Requires SHAP library.
        Loads the FINAL model and separates preprocessor/estimator.

        Args:
            model_base_path: Absolute base path (without .pkl) to the saved FINAL model.
            data_instance: A DataFrame containing a SINGLE row of data to explain.

        Returns:
            Dictionary containing SHAP values, base value, feature names, and instance data,
            or None on failure.
        """
        if not self.config.get("enable_prediction_explanation", False):
             logger.info("Prediction explanation disabled in config.")
             return None
        if shap is None:
            logger.error("SHAP library not installed. Cannot explain prediction.")
            return None
        if not model_base_path:
             logger.error("Explanation failed: model_base_path cannot be empty.")
             return None
        if data_instance is None or data_instance.empty or len(data_instance) != 1:
             logger.error("Explanation failed: data_instance must be a DataFrame with exactly one row.")
             return None

        logger.info(f"--- Explaining Prediction for Instance using model: {model_base_path} ---")
        explain_start_time = time.time()

        # 1. Load Model and get task type
        model_object, task_type = self._load_model_and_metadata(model_base_path)
        if model_object is None or task_type is None:
            return None # Error logged in helper

        # 2. Separate Preprocessor and Estimator from the loaded pipeline
        final_estimator = None
        preprocessor_pipeline = None
        transformed_instance_df = None # Store transformed data as DataFrame if possible

        if isinstance(model_object, Pipeline): # Check if it's a scikit-learn Pipeline
            try:
                if len(model_object.steps) > 1:
                    # Create pipeline with all steps EXCEPT the last one (preprocessor)
                    preprocessor_pipeline = Pipeline(model_object.steps[:-1])
                    final_estimator = model_object.steps[-1][1] # Estimator is last step
                    logger.info(f"Separated preprocessor and final estimator ({type(final_estimator).__name__}).")
                elif len(model_object.steps) == 1:
                     logger.warning("Loaded pipeline has only one step. Assuming it's the estimator.")
                     final_estimator = model_object.steps[0][1]
                     preprocessor_pipeline = None # No preprocessing steps
                else:
                     logger.error("Loaded pipeline has no steps.")
                     return None
            except Exception as e:
                logger.error(f"Failed to separate preprocessor/estimator: {e}", exc_info=True)
                return None
        else:
            # Assume loaded object is just the estimator (no PyCaret preprocessing pipeline)
            logger.warning("Loaded model is not a scikit-learn Pipeline. Assuming it's the estimator.")
            final_estimator = model_object
            preprocessor_pipeline = None

        # 3. Preprocess the single instance
        try:
            # Ensure input is DataFrame
            if not isinstance(data_instance, pd.DataFrame):
                logger.warning("Input data_instance is not DataFrame, attempting conversion.")
                data_instance = pd.DataFrame(data_instance, index=[0]) # Assume dict/Series like

            if preprocessor_pipeline:
                logger.info("Applying preprocessing steps to data instance...")
                # Transform data
                transformed_instance_array = preprocessor_pipeline.transform(data_instance)

                # Try to get feature names after transformation
                feature_names_out = None
                try:
                    # Use get_feature_names_out if available (Scikit-learn >= 1.0)
                    if hasattr(preprocessor_pipeline, 'get_feature_names_out'):
                        feature_names_out = preprocessor_pipeline.get_feature_names_out(input_features=data_instance.columns)
                    else: # Fallback for older sklearn or complex transformers
                         # Maybe infer from final step if it has feature names?
                         # Or try to get from setup_env? Needs setup_env to be loaded/available.
                         if self.setup_env and hasattr(self.setup_env, 'X_train'):
                              feature_names_out = self.setup_env.X_train.columns.tolist() # Use names from training
                              logger.warning("Using feature names from original training data for transformed SHAP data. This might be inaccurate if feature count changed.")
                              if len(feature_names_out) != transformed_instance_array.shape[1]:
                                   logger.error(f"Feature name count ({len(feature_names_out)}) mismatch with transformed data columns ({transformed_instance_array.shape[1]}). Cannot proceed reliably.")
                                   feature_names_out = None # Reset if mismatch

                except Exception as name_err:
                    logger.warning(f"Could not reliably determine feature names after transformation: {name_err}")

                # Create DataFrame from transformed array if possible
                if feature_names_out is not None and len(feature_names_out) == transformed_instance_array.shape[1]:
                    transformed_instance_df = pd.DataFrame(transformed_instance_array, columns=feature_names_out)
                else:
                    # Keep as array, SHAP might handle it but feature names will be generic
                    transformed_instance_df = transformed_instance_array # Store array directly
                    logger.warning("Could not create DataFrame with feature names for transformed data. SHAP feature names might be generic.")

                logger.info("Preprocessing complete.")
            else:
                logger.info("No preprocessing pipeline found. Using original data instance for SHAP.")
                transformed_instance_df = data_instance # Use original data

        except Exception as e:
            logger.error(f"Failed to apply preprocessing to data instance: {e}", exc_info=True)
            return None

        # 4. Initialize SHAP Explainer and Calculate Values
        if final_estimator is None:
             logger.error("Final estimator could not be determined.")
             return None
        if transformed_instance_df is None: # Can be DataFrame or NumPy array now
             logger.error("Transformed instance data is not available.")
             return None

        try:
            logger.info(f"Initializing SHAP explainer for: {type(final_estimator).__name__}")
            explainer = shap.Explainer(final_estimator, transformed_instance_df) # Without background data (simpler)

            logger.info(f"Calculating SHAP values using explainer: {type(explainer)}")
            # Pass the (potentially transformed) instance to the explainer
            shap_values = explainer(transformed_instance_df)
            logger.info("SHAP values calculated.")

            # 5. Format Output
            # Base value might be array for multi-output models
            base_value = shap_values.base_values[0] # Base value for the first instance
            if isinstance(base_value, (list, np.ndarray)):
                 base_value = base_value[0] # Take first output's base value if multi-output
                 logger.debug("SHAP base value is multi-output, using first value.")

            # SHAP values array shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
            instance_shap_values_array = shap_values.values[0] # Values for the first (only) instance
            if len(instance_shap_values_array.shape) > 1:
                 # Multi-class: shap_values[0].shape = (n_features, n_classes)
                 # For simplicity, explain the first class (index 0)
                 instance_shap_values_array = instance_shap_values_array[:, 0]
                 logger.warning("SHAP values are multi-output (multi-class); showing explanation for class 0.")

            # Get feature names (might be from SHAP object or inferred)
            feature_names = shap_values.feature_names
            if feature_names is None: # Fallback
                 if isinstance(transformed_instance_df, pd.DataFrame):
                     feature_names = transformed_instance_df.columns.tolist()
                 else: # If transformed data is array, use generic names
                     feature_names = [f"feature_{i}" for i in range(transformed_instance_df.shape[1])]

            explanation = {
                 "base_value": base_value, # Should be scalar now
                 "shap_values": instance_shap_values_array.tolist(), # Convert numpy array to list
                 "feature_names": feature_names,
                 "data_instance": data_instance.iloc[0].to_dict() # Original untransformed data row
            }

            logger.info(f"SHAP explanation duration: {time.time() - explain_start_time:.2f} seconds.")
            return explanation

        except Exception as e:
            logger.error(f"Failed during SHAP explanation calculation: {e}", exc_info=True)
            return None