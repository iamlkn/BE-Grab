from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
from typing import Optional, List, Dict, Any
import json # For loading/dumping if needed, though SQLAlchemy handles JSON type

from app.db.models.chatbot_sessions import ChatSessionState, JourneyLogEntry
import logging
# from app.db.models.datasets import Dataset # If needed for direct reference

SESSION_TIMEOUT_SECONDS_DB: int = 3600 * 24 # Example: 24 hours for DB persistence

logger = logging.getLogger(__name__)

def get_active_session_state(db: Session, dataset_id: int, session_uuid: str) -> Optional[ChatSessionState]:
    session = db.query(ChatSessionState)\
        .options(joinedload(ChatSessionState.journey_log_entries))\
        .filter(ChatSessionState.dataset_id == dataset_id, ChatSessionState.session_uuid == session_uuid)\
        .first()
    if session:
        session.last_accessed_at = func.now()
        db.commit()
        # db.refresh(session) # Only if you need to immediately see the updated last_accessed_at
    return session

def create_session_state(
    db: Session,
    dataset_id: int,
    session_uuid: str,
    initial_chat_history: Optional[List[Dict[str, Any]]] = None,
    auto_execute_enabled: bool = False
) -> ChatSessionState:
    db_session_state = ChatSessionState(
        dataset_id=dataset_id,
        session_uuid=session_uuid,
        chat_history_json=initial_chat_history if initial_chat_history else [],
        analysis_journey_log_json=[],
        auto_execute_enabled=auto_execute_enabled
        # Other fields will use defaults or be None
    )
    db.add(db_session_state)
    db.commit()
    db.refresh(db_session_state)
    return db_session_state

def update_session_state_from_service(
    db: Session,
    db_session_state: ChatSessionState, # The ORM object
    chat_history: List[Dict[str, Any]],
    analysis_journey_log: List[Dict[str, Any]],
    current_focus_filter: Optional[str],
    pending_code_json: Optional[Dict[str, Any]],
    pending_whatif_json: Optional[Dict[str, Any]],
    pending_focus_proposal_json: Optional[Dict[str, Any]],
    last_plot_path: Optional[str],
    auto_execute_enabled: bool # Persist this preference
) -> ChatSessionState:
    db_session_state.chat_history_json = chat_history
    db_session_state.analysis_journey_log_json = analysis_journey_log
    db_session_state.current_focus_filter = current_focus_filter
    db_session_state.pending_code_to_execute_json = pending_code_json
    db_session_state.pending_whatif_code_to_execute_json = pending_whatif_json
    db_session_state.pending_focus_proposal_json = pending_focus_proposal_json
    db_session_state.last_executed_plot_path = last_plot_path
    db_session_state.auto_execute_enabled = auto_execute_enabled
    # last_accessed_at is updated automatically by onupdate=func.now()
    
    db.commit()
    #db.refresh(db_session_state)
    return db_session_state

def add_journey_log_db_entry(
    db: Session,
    chat_session_state_orm: ChatSessionState, # Pass the parent ORM session state
    event_type: str,
    payload: Optional[Dict[str, Any]] = None
) -> JourneyLogEntry:
    db_log_entry = JourneyLogEntry(
        chat_session_state_id=chat_session_state_orm.id,
        event_type=event_type.upper(), # Good to standardize case
        payload_json=payload
    )
    db.add(db_log_entry)
    # The relationship ChatSessionState.journey_log_entries will automatically update
    # when the session is reloaded or if you append to it and commit.
    # For an immediate commit of the log entry:
    db.commit()
    db.refresh(db_log_entry) # To get its ID and timestamp from DB
    logger.info(f"Logged journey for session {chat_session_state_orm.session_uuid}: {event_type}")
    return db_log_entry

def delete_expired_session_states(db: Session) -> int:
    """
    Deletes session states where last_accessed_at is older than SESSION_TIMEOUT_SECONDS_DB.
    Returns the number of deleted sessions.
    NOTE: Date/time arithmetic is database-specific. This is a generic placeholder.
    For PostgreSQL: WHERE last_accessed_at < NOW() - INTERVAL 'X seconds'
    For SQLite: WHERE julianday('now') - julianday(last_accessed_at) * 86400.0 > X
    """
    # This is a simplified delete for example purposes.
    # A robust solution would use proper interval arithmetic for your DB.
    # For now, this won't actually delete based on time, just shows the structure.
    # You'd need to implement the correct WHERE clause.
    # Example for PostgreSQL (conceptual, assuming SESSION_TIMEOUT_SECONDS_DB is in seconds):
    # from sqlalchemy import text
    # result = db.execute(
    #     text("DELETE FROM chatbot_session_states WHERE last_accessed_at < NOW() - INTERVAL ':timeout seconds'").bindparams(timeout=SESSION_TIMEOUT_SECONDS_DB)
    # )
    # num_deleted = result.rowcount
    
    # For now, let's just log that this would be called.
    # Implement actual deletion based on your DB.
    num_deleted = 0 
    # query = db.query(ChatSessionState).filter(...) # Add your DB-specific time filter
    # num_deleted = query.delete(synchronize_session=False)
    # db.commit()
    logging.info(f"Placeholder for deleting expired sessions. Would delete {num_deleted} sessions.")
    return num_deleted

def get_sessions_for_dataset(db: Session, dataset_id: int) -> List[ChatSessionState]:
    """
    Retrieves all ChatSessionState records associated with a given dataset_id.
    Orders by last_accessed_at descending by default.
    """
    return db.query(ChatSessionState)\
        .filter(ChatSessionState.dataset_id == dataset_id)\
        .order_by(ChatSessionState.last_accessed_at.desc())\
        .all()
        
def get_latest_k_journey_log_entries(
    db: Session,
    chat_session_state_id: int, # Use the parent session's primary key
    k: int
) -> List[JourneyLogEntry]: # Or your FrontendJourneyLogEntry model name
    """
    Retrieves the latest k JourneyLogEntry records for a given chat_session_state_id,
    ordered by timestamp descending.
    """
    if k <= 0:
        return []
        
    return db.query(JourneyLogEntry)\
        .filter(JourneyLogEntry.chat_session_state_id == chat_session_state_id)\
        .order_by(JourneyLogEntry.timestamp.desc())\
        .limit(k)\
        .all()
        
def update_chat_name_by_ids(
    db: Session,
    dataset_id: int,
    session_uuid: str,
    new_chat_name: str
) -> Optional[ChatSessionState]:
    """
    Fetches a ChatSessionState by dataset_id and session_uuid, updates its chat_name, and returns it.
    """
    session = db.query(ChatSessionState)\
        .filter(ChatSessionState.dataset_id == dataset_id, ChatSessionState.session_uuid == session_uuid)\
        .first()
    if not session:
        return None
    session.chat_name = new_chat_name
    db.commit()
    db.refresh(session)
    logger.info(f"Updated chat_name for session {session_uuid} to '{new_chat_name}'")
    return session


def delete_session_state(db: Session, dataset_id: int, session_uuid: str) -> int:
    """
    Deletes a specific ChatSessionState identified by dataset_id and session_uuid.

    This function finds the session and deletes it directly. If the relationship
    to JourneyLogEntry has appropriate cascade options (like "all, delete-orphan"),
    related log entries will also be deleted.

    Args:
        db: The SQLAlchemy database session.
        dataset_id: The ID of the dataset associated with the session.
        session_uuid: The UUID of the session to delete.

    Returns:
        int: 1 if the session was successfully deleted, 0 otherwise (not found or error).
    """
    logger.info(f"Attempting to delete session: dataset_id={dataset_id}, session_uuid={session_uuid}")
    try:
        # Step 1: Find the specific session object
        session_to_delete = db.query(ChatSessionState)\
            .filter(ChatSessionState.dataset_id == dataset_id,
                    ChatSessionState.session_uuid == session_uuid)\
            .first() # Use first() to get one object or None

        # Step 2: Check if the session was found
        if session_to_delete:
            # Step 3: Delete the object using the session context
            db.delete(session_to_delete)

            # Step 4: Commit the transaction to make the deletion permanent
            db.commit()
            logger.info(f"Successfully deleted session: dataset_id={dataset_id}, session_uuid={session_uuid}")
            return 1 # Indicate one record was deleted
        else:
            # Session wasn't found, nothing to delete
            logger.warning(f"Session not found for deletion: dataset_id={dataset_id}, session_uuid={session_uuid}")
            return 0 # Indicate zero records were deleted

    except Exception as e:
        # Log any errors during the process
        logger.error(f"Error occurred while deleting session (dataset_id={dataset_id}, session_uuid={session_uuid}): {e}", exc_info=True)
        # Rollback the transaction in case of error to avoid partial changes
        db.rollback()
        return 0 