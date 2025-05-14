# --- Import from common.py ---
try:
    from .commons import StatusResponse, DataFrameStructure
except ImportError:
    print("Warning: Could not import schemas from common.py")
    # Define dummies if needed for other imports to work initially
    class StatusResponse: pass
    class DataFrameStructure: pass


# --- Import from automl_session.py ---
# Import all relevant schemas defined in that file
try:
    from .automl_sessions import (
        AutoMLSessionBase,
        AutoMLSessionCreate,
        AutoMLSessionStartStep1Request, # Schema for starting step 1
        AutoMLSessionStep1Response,
        AutoMLSessionStep1Result,       # Schema for step 1 results (used in Response)
        AutoMLSessionStartStep2Request, # Schema for starting step 2
        AutoMLSessionStep2Result,       # Schema for step 2 results
        AutoMLSessionStartStep3Request, # Schema for starting step 3
        AutoMLSessionStep3Result,       # Schema for step 3 results
        AutoMLSessionResponse,          # The main response schema showing all steps
        AutoMLSessionErrorResponse,
        AutoMLSessionUpdateStepStatus   # Internal schema for updates (optional export)
    )
except ImportError:
    print("Warning: Could not import schemas from automl_session.py")


# --- Import from finalized_models.py ---
# Import schemas related to the FinalizedModel table
try:
    from .finalized_models import (
        FinalizedModelResponse
        
        # Add FinalizedModelCreate/Update schemas here if you defined them
    )
except ImportError:
    print("Warning: Could not import schemas from finalized_models.py")

# --- Import from predictions.py ---
# Import schemas related to the prediction
try:
    from .predictions import PredictionRequest, PredictionResponse
except ImportError:
    print("Warning: Could not import schemas from predictions.py")