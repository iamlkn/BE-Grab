from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class ChatbotQueryRequest(BaseModel):
    query: str = Field(..., description="The user's query or command for the chatbot.")
    # session_id: Optional[str] = None # We'll primarily use Header/Query for session_id in API path

class ChatbotResponseItem(BaseModel):
    type: str = Field(..., description="Type of the response item.")
    message: Optional[str] = None
    content: Optional[Union[str, List[str], Dict[str, Any]]] = None

    # Code related
    code: Optional[str] = None
    code_attempted: Optional[str] = None
    is_plot_code: Optional[bool] = None

    # File/Path related
    path: Optional[str] = None

    # Execution/Output related
    output: Optional[str] = None
    last_error: Optional[str] = None

    # AI Interaction specific
    explanation: Optional[str] = None
    insights: Optional[str] = None
    questions: Optional[List[str]] = None
    ai_review: Optional[str] = None

    # Journey/History
    log: Optional[List[Dict[str, Any]]] = None

    # UI/UX flow
    next_actions: Optional[List[str]] = None
    attempt_number: Optional[int] = None

    # Contextual info
    original_query: Optional[str] = None
    filter_condition: Optional[str] = None

    # /focus specific
    impact_assessment: Optional[str] = None

    class Config:
        # from_attributes = True # For Pydantic V2
        pass

class ChatbotInteractionResponse(BaseModel):
    dataset_id: int
    session_id: str # Now required, server will ensure it's set
    responses: List[ChatbotResponseItem]

class NewSessionResponse(BaseModel):
    dataset_id: int
    session_id: str
    message: str
    
class JourneyLogEntryResponse(BaseModel):
    timestamp: datetime
    event_type: str
    payload_json: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
        
class FrontendJourneyLogItem(BaseModel): # A new schema for the API output
    timestamp: datetime
    event_type: str
    payload: Optional[Dict[str, Any]] # This is what the FE will see
        
class LoadedSessionResponse(BaseModel): # Schema for full session state reload
    dataset_id: int
    session_id: str
    #chat_history_json: List[Dict[str, Any]] # The AI-compatible history
    #analysis_journey_log_json: List[Dict[str, Any]]
    journey_log: List[FrontendJourneyLogItem] # The rich journey log for UI reconstruction
    current_focus_filter: Optional[str]
    #pending_code_to_execute_json: Optional[Dict[str, Any]]
    #pending_whatif_code_to_execute_json: Optional[Dict[str, Any]] # Add if you have this state
    #pending_focus_proposal_json: Optional[Dict[str, Any]] # Add if you have this state
    #last_executed_plot_path: Optional[str] # This is the path
    #auto_execute_enabled: bool
    
class SessionInfo(BaseModel):
    session_uuid: str
    created_at: datetime
    last_accessed_at: datetime
    chat_name: Optional[str]

    class Config:
        from_attributes = True

class SessionListResponse(BaseModel):
    dataset_id: int
    sessions: List[SessionInfo]
    
class DetailResponse(BaseModel):
    detail: str
    
class ChatSessionUpdateName(BaseModel):
    """Schema for updating the chat session name."""
    chat_name: Optional[str] = Field(
        default=None, # Allow setting name to None (or empty string, adjust validation if needed)
        max_length=255,
        description="The new name for the chat session. Null or empty string might be allowed depending on business logic."
    )

class ChatSessionResponse(BaseModel):
    """Schema for returning chat session state details."""
    id: int
    dataset_id: int
    session_uuid: str
    chat_name: Optional[str] = None
    auto_execute_enabled: bool
    created_at: datetime
    last_accessed_at: datetime