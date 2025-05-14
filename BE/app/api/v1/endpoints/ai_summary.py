import base64
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body
from typing import Optional, Annotated
import json # For parsing JSON string from form data

from app.services.ai_summary_service import ai_summary_service_instance, encode_image_to_base64
from app.schemas.ai_summary import (
    AISummaryResponse,
    SummaryStatsInput,
    CorrelationMatrixInput,
    ModelPerformanceInput,
    TunedModelInput,
    TunedModelResultsData,
    BaselineModelMetricsData
)

import os
import base64

router = APIRouter(
    prefix='/v1',
    tags=['ai_summary']
)

@router.post("/summary-stats", response_model=AISummaryResponse)
async def get_summary_statistics_analysis(
    payload: SummaryStatsInput
):
    try:
        summary = ai_summary_service_instance.get_ai_summary(
            data=payload.data,
            input_type='summary_stats'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='summary_stats'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlation-matrix", response_model=AISummaryResponse)
async def get_correlation_matrix_analysis(
    payload: CorrelationMatrixInput
):
    try:
        summary = ai_summary_service_instance.get_ai_summary(
            data=payload.data,
            input_type='correlation_matrix'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='correlation_matrix'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model-performance", response_model=AISummaryResponse)
async def get_model_performance_analysis(
    payload: ModelPerformanceInput
):
    try:
        summary = ai_summary_service_instance.get_ai_summary(
            data=payload.model_dump(), # Pass the whole dict
            input_type='model_performance'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='model_performance'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tuned-model-evaluation", response_model=AISummaryResponse)
async def get_tuned_model_evaluation(
    tuning_data_json: Annotated[str, Form()],
    feature_importance_image_path: Annotated[Optional[str], Form()] = None
):
    try:
        try:
            tuning_data_dict = json.loads(tuning_data_json)
            validated_tuning_data = TunedModelResultsData(**tuning_data_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for tuning_data_json.")
        except Exception as pydantic_error:
            raise HTTPException(status_code=422, detail=f"Invalid tuning_data structure: {pydantic_error}")

        service_payload = {"tuning_data": validated_tuning_data.model_dump()}

        if feature_importance_image_path:
            # Validate if the path exists (basic check)
            if not os.path.exists(feature_importance_image_path):
                raise HTTPException(status_code=400, detail=f"Image path does not exist: {feature_importance_image_path}")
            if not os.path.isfile(feature_importance_image_path):
                 raise HTTPException(status_code=400, detail=f"Image path is not a file: {feature_importance_image_path}")

            # Determine MIME type (basic inference)
            file_ext = os.path.splitext(feature_importance_image_path)[1].lower()
            mime_type = "image/png" if file_ext == ".png" else \
                        "image/jpeg" if file_ext in [".jpg", ".jpeg"] else \
                        "image/gif" if file_ext == ".gif" else \
                        "image/webp" if file_ext == ".webp" else \
                        "application/octet-stream" # Fallback

            if mime_type == "application/octet-stream" and file_ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                # Potentially raise an error or warning for unsupported inferred types
                print(f"Warning: Could not reliably determine MIME type for {feature_importance_image_path}, falling back to octet-stream.")
                # Consider raising HTTPException if strict type checking is required.

            encoded_image = encode_image_to_base64(feature_importance_image_path)
            if encoded_image:
                service_payload["image_base64"] = encoded_image
                service_payload["image_mime_type"] = mime_type
            else:
                # encode_image_to_base64 prints its own warning if file not found or encoding error
                # The service will proceed without the image if encoding fails
                print(f"Warning: Could not encode image from path: {feature_importance_image_path}")
        # If no path is provided, the service will handle it

        summary = ai_summary_service_instance.get_ai_summary(
            data=service_payload,
            input_type='tuned_model_with_image_eval'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='tuned_model_with_image_eval'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.post("/baseline-model-evaluation", response_model=AISummaryResponse)
async def get_baseline_model_evaluation(
    metrics_data_json: Annotated[str, Form()], # JSON string for BaselineModelMetricsData
    feature_importance_image_path: Annotated[Optional[str], Form()] = None
):
    try:
        try:
            metrics_dict = json.loads(metrics_data_json)
            # Validate with Pydantic model
            validated_metrics_data = BaselineModelMetricsData(**metrics_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for metrics_data_json.")
        except Exception as pydantic_error: # Catches Pydantic validation errors
            raise HTTPException(status_code=422, detail=f"Invalid metrics_data structure: {pydantic_error}")

        service_payload = {"metrics_data": validated_metrics_data.model_dump(by_alias=True)}

        if feature_importance_image_path:
            if not os.path.exists(feature_importance_image_path):
                raise HTTPException(status_code=400, detail=f"Image path does not exist: {feature_importance_image_path}")
            if not os.path.isfile(feature_importance_image_path):
                 raise HTTPException(status_code=400, detail=f"Image path is not a file: {feature_importance_image_path}")

            file_ext = os.path.splitext(feature_importance_image_path)[1].lower()
            mime_type = "image/png" if file_ext == ".png" else \
                        "image/jpeg" if file_ext in [".jpg", ".jpeg"] else \
                        "image/gif" if file_ext == ".gif" else \
                        "image/webp" if file_ext == ".webp" else \
                        "application/octet-stream"
            
            if mime_type == "application/octet-stream" and file_ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                 print(f"Warning: Could not reliably determine MIME type for {feature_importance_image_path}, falling back to octet-stream.")


            encoded_image = encode_image_to_base64(feature_importance_image_path)
            if encoded_image:
                service_payload["image_base64"] = encoded_image
                service_payload["image_mime_type"] = mime_type
            else:
                print(f"Warning: Could not encode image from path: {feature_importance_image_path}")

        summary = ai_summary_service_instance.get_ai_summary(
            data=service_payload,
            input_type='baseline_model_with_image_eval'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)

        return AISummaryResponse(
            summary_html=summary,
            input_type='baseline_model_with_image_eval'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")