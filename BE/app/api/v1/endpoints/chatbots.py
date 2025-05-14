from fastapi import APIRouter, Depends, HTTPException, Header, Query, BackgroundTasks, status, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Tuple, Optional
import time
import uuid # For generating session IDs
import logging
import os
import random # For opportunistic background task scheduling
from app.api.v1.dependencies import get_db
from app.crud.datasets import crud_dataset
from app.crud import chatbot_sessions as crud_chatbot_session
from app.schemas import chatbots as chatbot_schemas
from app.services.chatbot_service import ChatbotService #ChatbotService
from app.db.models.chatbot_sessions import ChatSessionState, JourneyLogEntry
DEFAULT_LATEST_K_JOURNEY_LOGS = 1

router = APIRouter()
logger = logging.getLogger(__name__)

# --- DB-Backed Session Management Helper Functions ---
def _get_or_create_db_session(
    dataset_id: int,
    session_uuid: str,
    db: Session
) -> Tuple[ChatbotService, ChatSessionState]: # Return both service and DB ORM object
    """
    Retrieves ChatbotService and its DB state for a session, or creates them.
    """
    db_session_state = crud_chatbot_session.get_active_session_state(db, dataset_id, session_uuid)
    db_dataset = None # To store dataset info

    if db_session_state:
        logger.info(f"Found existing DB session: {session_uuid} for dataset_id: {dataset_id}")
        db_dataset = db_session_state.dataset # Assumes relationship is loaded or use db_dataset.id
        if not db_dataset: # Should not happen if FK constraint is good
             db_dataset = crud_dataset.get(db, id=dataset_id)
             if not db_dataset:
                  raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} for existing session {session_uuid} not found.")
        
        # Hydrate service with state from DB
        try:
            # Safely parse JSON fields, providing defaults if None or invalid
            chat_history = db_session_state.chat_history_json if db_session_state.chat_history_json else []
            journey_log = db_session_state.analysis_journey_log_json if db_session_state.analysis_journey_log_json else []
            pending_code = db_session_state.pending_code_to_execute_json
            pending_whatif = db_session_state.pending_whatif_code_to_execute_json
            pending_focus_proposal = db_session_state.pending_focus_proposal_json

            service = ChatbotService(
                csv_file_path=db_dataset.file_path,
                initial_chat_history=chat_history,
                initial_journey_log=journey_log,
                initial_focus_filter=db_session_state.current_focus_filter,
                initial_pending_code=pending_code,
                initial_pending_whatif=pending_whatif,
                initial_pending_focus_proposal=pending_focus_proposal,
                initial_last_plot_path=db_session_state.last_executed_plot_path,
                initial_auto_execute_enabled=db_session_state.auto_execute_enabled
            )
        except Exception as e:
            logger.error(f"Error hydrating ChatbotService from DB state for session {session_uuid}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error loading session state.")
    else:
        logger.info(f"Creating new DB session: {session_uuid} for dataset_id: {dataset_id}")
        db_dataset = crud_dataset.get(db, id=dataset_id)
        if not db_dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found for new session.")
        if not db_dataset.file_path or not os.path.exists(db_dataset.file_path):
            raise HTTPException(status_code=500, detail=f"File path for dataset {dataset_id} invalid.")

        # Create a fresh service to get its initial history (system prompt)
        temp_service_for_init = ChatbotService(csv_file_path=db_dataset.file_path)
        initial_persistable_state = temp_service_for_init.get_current_state_for_persistence()
        initial_serializable_history = initial_persistable_state["chat_history"]
        
        db_session_state = crud_chatbot_session.create_session_state(
            db, dataset_id, session_uuid, initial_chat_history=initial_serializable_history, auto_execute_enabled=True
        )
        # Now create the service instance for this new session using the just-created history
        service = ChatbotService(
            csv_file_path=db_dataset.file_path,
            initial_chat_history=initial_serializable_history, # from db_session_state.chat_history_json or temp_service
            initial_auto_execute_enabled=db_session_state.auto_execute_enabled # from DB default
            # other initial states will be None/empty list by default in ChatbotService
        )
    
    return service, db_session_state

def _db_cleanup_expired_sessions(db: Session = Depends(get_db)): # Make it a dependency for background task
    """Background task to delete expired sessions from DB."""
    try:
        num_deleted = crud_chatbot_session.delete_expired_session_states(db)
        if num_deleted > 0:
            logger.info(f"DB Background cleanup: Deleted {num_deleted} expired sessions.")
        else:
            logger.debug("DB Background cleanup: No expired sessions found to delete.")
    except Exception as e:
        logger.error(f"Error during DB background session cleanup: {e}", exc_info=True)
        
# --- Dependency for Session Management with DB ---
class DBSessionManager:
    def __init__(self, dataset_id: int,
                 session_id_header: Optional[str] = Header(None, alias="X-Session-ID"),
                 session_id_query: Optional[str] = Query(None, alias="session_id")):
        self.dataset_id = dataset_id
        self.session_id_header = session_id_header
        self.session_id_query = session_id_query
        self.resolved_session_uuid: Optional[str] = None
        self.db_session_state_orm: Optional[ChatSessionState] = None # To store the ORM object

    def _resolve_session_uuid(self, generate_if_missing: bool = False) -> str:
        if self.resolved_session_uuid:
            return self.resolved_session_uuid

        session_id = self.session_id_header or self.session_id_query
        if not session_id and generate_if_missing:
            session_id = str(uuid.uuid4())
            logger.info(f"No session_id provided for dataset {self.dataset_id}, generated new UUID: {session_id}")
        elif not session_id and not generate_if_missing:
            raise HTTPException(
                status_code=400,
                detail="Session ID (UUID) is required. Please provide it via 'X-Session-ID' header or 'session_id' query parameter, or start a new session."
            )
        self.resolved_session_uuid = session_id
        return session_id

    async def get_service_and_db_state(self, db: Session = Depends(get_db)) -> Tuple[ChatbotService, ChatSessionState]:
        session_uuid = self._resolve_session_uuid(generate_if_missing=False)
        service, db_state = _get_or_create_db_session(self.dataset_id, session_uuid, db)
        self.db_session_state_orm = db_state # Store for later saving
        return service, db_state

    async def start_new_session(self, db: Session = Depends(get_db)) -> Tuple[ChatbotService, ChatSessionState, str]:
        """Generates a new session UUID and creates/gets the service and DB state."""
        new_session_uuid = str(uuid.uuid4()) # Always generate a new one here
        self.resolved_session_uuid = new_session_uuid # Store it for consistency if manager instance is reused (unlikely here)
        logger.info(f"DBSessionManager: Creating new session service with explicitly generated UUID: {new_session_uuid} for dataset {self.dataset_id}")
        
        service, db_state = _get_or_create_db_session(self.dataset_id, new_session_uuid, db)
        self.db_session_state_orm = db_state
        return service, db_state, new_session_uuid

# --- API Endpoints ---
@router.post(
    "/sessions/start/{dataset_id}",
    response_model=chatbot_schemas.NewSessionResponse,
    summary="Start a new chatbot session for a dataset (DB backed)",
    tags=["chatbot_session_db"]
)
async def start_new_db_chatbot_session(
    dataset_id: int, # Path parameter
    chat_name: Optional[str],
    db: Session = Depends(get_db)
):
    """
    Explicitly starts a new chat session for the given dataset_id.
    A new unique session_uuid will be generated and returned.
    Use this session_uuid in subsequent calls to `/interact/{dataset_id}`.
    """
    # 1. Validate dataset_id
    db_dataset = crud_dataset.get(db, id=dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")

    # 2. Generate a new session_uuid
    new_session_uuid = str(uuid.uuid4())
    logger.info(f"Starting new explicit DB session {new_session_uuid} for dataset {dataset_id}")

    # 3. Call the core logic to create the session in DB and get the service
    #    _get_or_create_db_session will handle creating the ChatSessionState in the DB
    #    and initializing the service (which includes getting the initial system prompt history).
    try:
        # This call will create the ChatSessionState in the DB if it doesn't exist
        # (which it won't for a new_session_uuid).
        _service, _db_state = _get_or_create_db_session(dataset_id, new_session_uuid, db)
        
        # Log SESSION_START
        crud_chatbot_session.add_journey_log_db_entry(
            db, _db_state, "SESSION_START", {"message": "New session initiated."}
        )
        # Log initial AI acknowledgment
        ai_history = _service.get_persistable_ai_history() # Get AI-compatible history
        if ai_history and len(ai_history) > 1 and ai_history[1].get("role") == "model":
            try:
                ack_text = ai_history[1].get("parts", [{}])[0].get("text", "")
                crud_chatbot_session.add_journey_log_db_entry(
                    db, _db_state, "AI_INITIAL_ACK", {"content": ack_text} # 'content' to match ChatbotResponseItem
                )
            except Exception as e:
                logger.warning(f"Could not log AI initial ack: {e}")
        
        crud_chatbot_session.update_chat_name_by_ids(db, dataset_id, new_session_uuid, chat_name)

        
    except HTTPException as e: # Catch specific errors from _get_or_create_db_session
        raise e
    except Exception as e:
        logger.error(f"Failed to create new session objects for {new_session_uuid}, dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize new session.")

    return chatbot_schemas.NewSessionResponse(
        dataset_id=dataset_id,
        session_id=new_session_uuid, # This is the session_uuid
        message="New DB-backed session started. Use the provided session_id for interactions."
    )


@router.post(
    "/interact/{dataset_id}",
    response_model=chatbot_schemas.ChatbotInteractionResponse,
    summary="Interact with the Chatbot (DB backed session)",
    tags=["chatbot_interaction_db"]
)
async def interact_with_db_chatbot(
    dataset_id: int, # Path param, used by DBSessionManager
    query_request: chatbot_schemas.ChatbotQueryRequest,
    background_tasks: BackgroundTasks,
    session_manager: DBSessionManager = Depends(), # Dependency
    db: Session = Depends(get_db) # For saving state
):
    service, db_session_state_orm = await session_manager.get_service_and_db_state(db=db) # Pass db here
    current_session_uuid = session_manager.resolved_session_uuid

    if not current_session_uuid: # Should be caught by DBSessionManager
        raise HTTPException(status_code=400, detail="Session UUID could not be resolved.")

    try:
        # Log USER_QUERY to journey
        crud_chatbot_session.add_journey_log_db_entry(
            db, db_session_state_orm, "USER_QUERY", {"content": query_request.query} # Using 'content' for consistency
        )
        
        service_interaction_item_dicts = service.process_user_query(user_query=query_request.query)
        # Log each service interaction item to journey log
        for item_dict in service_interaction_item_dicts:
            event_type = item_dict.get("type", "UNKNOWN_SERVICE_ITEM") # Get 'type' from item_dict
            payload = {k: v for k, v in item_dict.items() if k != "type"} # Payload is the rest of the dict
            crud_chatbot_session.add_journey_log_db_entry(
                db, db_session_state_orm, event_type, payload
            )
        
        raw_responses = service.process_user_query(user_query=query_request.query)

        # After processing, get the current state from the service and save it to DB
        current_service_state = service.get_current_state_for_persistence()
        crud_chatbot_session.update_session_state_from_service(
            db=db,
            db_session_state=db_session_state_orm, # Pass the ORM object
            chat_history=current_service_state["chat_history"],
            analysis_journey_log=current_service_state["analysis_journey_log"],
            current_focus_filter=current_service_state["current_focus_filter"],
            pending_code_json=current_service_state["pending_code_to_execute"],
            pending_whatif_json=current_service_state["pending_whatif_code_to_execute"],
            pending_focus_proposal_json=current_service_state["pending_focus_proposal"],
            last_plot_path=current_service_state["last_executed_plot_path"],
            auto_execute_enabled=current_service_state["auto_execute_enabled"]
        )

        formatted_responses: List[chatbot_schemas.ChatbotResponseItem] = []
        for res_item_dict in raw_responses:
            try:
                formatted_responses.append(chatbot_schemas.ChatbotResponseItem(**res_item_dict))
            except Exception as pydantic_error:
                logger.error(f"Pydantic validation error for response item: {res_item_dict}. Error: {pydantic_error}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal error formatting response: {pydantic_error}")
        
        if random.randint(1, 50) == 1: # Less frequent for DB cleanup
            logger.debug("Scheduling background DB session cleanup.")
            background_tasks.add_task(_db_cleanup_expired_sessions, db=db) # Pass db to background task if it needs it

        return chatbot_schemas.ChatbotInteractionResponse(
            dataset_id=dataset_id,
            session_id=current_session_uuid,
            responses=formatted_responses
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during DB chatbot interaction for dataset {dataset_id}, session {current_session_uuid}: {e}", exc_info=True)
        # Attempt to save state even on error, if possible (though state might be inconsistent)
        # This is optional and depends on how you want to handle partial failures.
        # For now, we won't save on generic exception to avoid persisting bad state.
        raise HTTPException(status_code=500, detail=f"An internal error occurred during chatbot interaction: {str(e)}")

@router.get(
    "/sessions/{dataset_id}/state",
    response_model=chatbot_schemas.LoadedSessionResponse, # Use your new schema
    tags=["chatbot_session_db"]
)
async def get_db_session_state_endpoint( # Renamed function
    dataset_id: int,
    session_manager: DBSessionManager = Depends(),
    db: Session = Depends(get_db)
):
    session_uuid = session_manager._resolve_session_uuid(generate_if_missing=False)
    # get_active_session_state should eagerly load journey_log_entries
    db_session_orm = crud_chatbot_session.get_active_session_state(db, dataset_id, session_uuid)
    if not db_session_orm:
        raise HTTPException(status_code=404, detail="Session not found.")

    journey_log_for_fe = []
    for entry in db_session_orm.journey_log_entries:
        journey_log_for_fe.append(
            chatbot_schemas.FrontendJourneyLogItem( # Map manually
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                payload=entry.payload_json # Map from payload_json to payload
            )
        )

    return chatbot_schemas.LoadedSessionResponse(
        dataset_id=db_session_orm.dataset_id,
        session_id=db_session_orm.session_uuid,
        #chat_history_json=db_session_orm.chat_history_json or [], # Send the AI history
        journey_log=journey_log_for_fe, # Send the formatted journey log
        #analysis_journey_log_json=db_session_orm.analysis_journey_log_json or [],
        current_focus_filter=db_session_orm.current_focus_filter,
        #pending_code_to_execute_json=db_session_orm.pending_code_to_execute_json,
        #pending_whatif_code_to_execute_json=db_session_orm.pending_whatif_code_to_execute_json, # Add if present
        #pending_focus_proposal_json=db_session_orm.pending_focus_proposal_json, # Add if present
        #last_executed_plot_path=db_session_orm.last_executed_plot_path,
        #auto_execute_enabled=db_session_orm.auto_execute_enabled
    )
    
@router.get(
    "/sessions/{dataset_id}/state/latest",
    response_model=chatbot_schemas.LoadedSessionResponse,
    tags=["chatbot_session_db"]
)
async def get_db_session_k_state_endpoint(
    dataset_id: int,
    session_manager: DBSessionManager = Depends(),
    db: Session = Depends(get_db),
    k: int = Query(DEFAULT_LATEST_K_JOURNEY_LOGS, ge=1, le=100, description="Number of latest journey log entries to retrieve.") # Optional query param k
):
    session_uuid = session_manager._resolve_session_uuid(generate_if_missing=False)

    db_session_orm = crud_chatbot_session.get_active_session_state(db, dataset_id, session_uuid)
    
    if not db_session_orm:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")

    # Fetch the latest k journey log entries using the new CRUD function
    latest_k_journey_entries_orm = []
    if k and k > 0: # Only fetch if k is provided and positive
        latest_k_journey_entries_orm = crud_chatbot_session.get_latest_k_journey_log_entries(
            db, chat_session_state_id=db_session_orm.id, k=k
        )
        latest_k_journey_entries_orm.reverse()


    journey_log_for_fe = []
    for entry in latest_k_journey_entries_orm:
        journey_log_for_fe.append(
            chatbot_schemas.FrontendJourneyLogItem( # Map manually
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                payload=entry.payload_json # Map from payload_json to payload
            )
        )

    return chatbot_schemas.LoadedSessionResponse(
        dataset_id=db_session_orm.dataset_id,
        session_id=db_session_orm.session_uuid,
        journey_log=journey_log_for_fe, # This now contains the latest k entries (oldest of them first)
        current_focus_filter=db_session_orm.current_focus_filter,
    )


@router.get(
    "/starter-questions/{dataset_id}",
    response_model=List[str],
    summary="Get starter questions for a dataset (stateless call)",
    tags=["chatbot_utils_db"] # Changed tag slightly
)
async def get_db_starter_questions_for_dataset( # Renamed function slightly
    dataset_id: int,
    db: Session = Depends(get_db)
):
    # This endpoint remains stateless as before
    db_dataset = crud_dataset.get(db, id=dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    if not db_dataset.file_path or not os.path.exists(db_dataset.file_path):
        raise HTTPException(status_code=500, detail="Dataset file path invalid.")
    try:
        temp_service = ChatbotService(csv_file_path=db_dataset.file_path)
        return temp_service.get_starter_questions()
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Error init temp service: {ve}")
    except Exception as e:
        logger.error(f"Error getting starter questions (DB) for dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting starter questions: {e}")


@router.post(
    "/sessions/{dataset_id}/clear-history",
    summary="Clears chat history and state for a DB-backed session",
    status_code=204,
    tags=["chatbot_session_db"]
)
async def clear_db_session_history(
    dataset_id: int, # Path param for DBSessionManager
    session_manager: DBSessionManager = Depends(),
    db: Session = Depends(get_db)
):
    # session_manager will resolve session_uuid from header/query
    # It will raise HTTPException if session_uuid is not provided
    session_uuid = session_manager._resolve_session_uuid(generate_if_missing=False)

    db_session_state = crud_chatbot_session.get_active_session_state(db, dataset_id, session_uuid)

    if db_session_state:
        logger.info(f"Clearing history and state for DB session: {session_uuid}, dataset_id: {dataset_id}")
        db.query(JourneyLogEntry).filter(JourneyLogEntry.chat_session_state_id == db_session_state.id).delete(synchronize_session='fetch')
        db_dataset = db_session_state.dataset
        if not db_dataset or not db_dataset.file_path or not os.path.exists(db_dataset.file_path):
            logger.error(f"Cannot clear session {session_uuid}: dataset or its file path is invalid.")
            raise HTTPException(status_code=500, detail="Cannot reset session due to invalid dataset information.")

        temp_service_for_init = ChatbotService(csv_file_path=db_dataset.file_path)
        initial_history = temp_service_for_init.chat_session.history if temp_service_for_init.chat_session else []
        
        crud_chatbot_session.add_journey_log_db_entry(
            db, db_session_state, "SESSION_CLEARED", {"message": "Session state and journey log reset."}
        )
        
        crud_chatbot_session.update_session_state_from_service(
            db=db,
            db_session_state=db_session_state,
            chat_history=initial_history,
            analysis_journey_log=[],
            current_focus_filter=None,
            pending_code_json=None,
            pending_whatif_json=None,
            pending_focus_proposal_json=None,
            last_plot_path=None,
            auto_execute_enabled=False # Reset to default, or load from user preference if you add that
        )
        return # 204 No Content
    else:
        logger.info(f"Attempted to clear history for non-existent/timed-out DB session: {session_uuid}")
        return # Idempotent if session doesn't exist
    
@router.get(
    "/sessions/list/{dataset_id}", # Using "list" in the path for clarity
    response_model=chatbot_schemas.SessionListResponse,
    summary="List all chatbot sessions for a given dataset",
    tags=["chatbot_sessions"] # Use your existing tag or a new one for session management
)
async def list_sessions_for_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieves a list of all session UUIDs and their metadata
    associated with the provided `dataset_id`.
    """
    # First, verify the dataset itself exists (optional, but good practice)
    db_dataset = crud_dataset.get(db, id=dataset_id) # Use your actual get_dataset method
    if not db_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")

    db_sessions = crud_chatbot_session.get_sessions_for_dataset(db, dataset_id=dataset_id)

    if not db_sessions:
        # Return an empty list if no sessions found, rather than 404,
        # as it's valid for a dataset to have no sessions yet.
        return chatbot_schemas.SessionListResponse(dataset_id=dataset_id, sessions=[])

    # Convert ORM objects to Pydantic schema objects
    session_infos = [chatbot_schemas.SessionInfo.from_orm(session) for session in db_sessions]
    
    return chatbot_schemas.SessionListResponse(
        dataset_id=dataset_id,
        sessions=session_infos
    )
    
@router.delete(
    "/datasets/{dataset_id}/sessions/{session_uuid}",
    # response_model=DetailResponse, # Use if returning the DetailResponse model
    status_code=status.HTTP_200_OK, # Can also use 204 No Content if preferred
    summary="Delete a specific chat session",
    tags=["Chat Sessions"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Chat session not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during deletion"},
    }
)
def delete_chat_session(
    *,
    db: Session = Depends(get_db), # Inject DB session
    dataset_id: int,                   # Get dataset_id from path
    session_uuid: str                  # Get session_uuid from path
) -> chatbot_schemas.DetailResponse: # Specify the return type hint
    """
    Deletes the chat session state identified by the dataset ID and session UUID.

    If the session exists, it will be deleted from the database.
    If the underlying database relationship has cascade delete configured for
    associated JourneyLogEntry records, they will also be removed.
    """
    logger.info(f"Received request to delete session: dataset_id={dataset_id}, session_uuid={session_uuid}")

    deleted_count = crud_chatbot_session.delete_session_state(
        db=db,
        dataset_id=dataset_id,
        session_uuid=session_uuid
    )

    if deleted_count == 0:
        # The CRUD function handles logging the "not found" case internally,
        # but we raise the HTTP exception here for the client.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session not found for dataset_id {dataset_id} and session_uuid {session_uuid}"
        )
    elif deleted_count == 1:
         # Successfully deleted
        logger.info(f"API endpoint confirmed deletion for session: dataset_id={dataset_id}, session_uuid={session_uuid}")
        # Return a success message (adjust if you prefer 204 No Content)
        return chatbot_schemas.DetailResponse(detail="Chat session deleted successfully")
    else:
        # This case shouldn't happen with the current CRUD logic (returns 0 or 1),
        # but good practice to handle unexpected return values.
        logger.error(f"Unexpected return value from delete_session_state: {deleted_count}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during session deletion."
        )
        
@router.patch(
    "/datasets/{dataset_id}/sessions/{session_uuid}", # Use PATCH on the session resource URL
    response_model=chatbot_schemas.ChatSessionResponse, # Specify the response schema
    status_code=status.HTTP_200_OK,
    summary="Update chat session name",
    tags=["Chat Sessions"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Chat session not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error in request body"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during update"},
    }
)
def update_chat_session_name(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    session_uuid: str,
    update_data: chatbot_schemas.ChatSessionUpdateName = Body(...) # Get data from request body, parsed into the schema
) -> chatbot_schemas.ChatSessionResponse:
    """
    Updates the 'chat_name' for a specific chat session identified by
    dataset ID and session UUID.
    """
    logger.info(f"Received request to update chat_name for session: dataset_id={dataset_id}, session_uuid={session_uuid}")

    # Use the existing CRUD function
    updated_session = crud_chatbot_session.update_chat_name_by_ids(
        db=db,
        dataset_id=dataset_id,
        session_uuid=session_uuid,
        new_chat_name=update_data.chat_name # Pass the name from the request body schema
    )

    if updated_session is None:
        # CRUD function returns None if session wasn't found
        logger.warning(f"Session not found for name update: dataset_id={dataset_id}, session_uuid={session_uuid}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session not found for dataset_id {dataset_id} and session_uuid {session_uuid}"
        )

    logger.info(f"Successfully updated chat_name for session: dataset_id={dataset_id}, session_uuid={session_uuid} to '{update_data.chat_name}'")
    return updated_session