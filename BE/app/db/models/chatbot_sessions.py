from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Text, Boolean, event
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base # Your SQLAlchemy Base

class ChatSessionState(Base):
    __tablename__ = "chatbot_session_states"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    session_uuid = Column(String, unique=True, index=True, nullable=False) # The client-facing session ID
    chat_name = Column(String, nullable=True)
    
    # Store complex Python objects as JSON strings or use database-native JSON types
    chat_history_json = Column(JSON, nullable=True) # Stores the list of history dicts
    analysis_journey_log_json = Column(JSON, nullable=True) # Stores the list of journey log dicts
    
    current_focus_filter = Column(String, nullable=True)
    pending_code_to_execute_json = Column(JSON, nullable=True)
    pending_whatif_code_to_execute_json = Column(JSON, nullable=True)
    pending_focus_proposal_json = Column(JSON, nullable=True)
    last_executed_plot_path = Column(String, nullable=True)
    auto_execute_enabled = Column(Boolean, default=True) # Per-session auto-execute preference

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    dataset = relationship("Dataset")
    # Relationship to JourneyLogEntry table
    journey_log_entries = relationship(
        "JourneyLogEntry",
        back_populates="session_state",
        cascade="all, delete-orphan",
        order_by="JourneyLogEntry.timestamp"
    )
    
@event.listens_for(ChatSessionState, "after_insert")
def set_chat_name(mapper, connection, target):
    if not target.chat_name:
        new_chat_name = f"Chat {target.id}"
        connection.execute(
            ChatSessionState.__table__.update()
            .where(ChatSessionState.id == target.id)
            .values(chat_name=new_chat_name)
        )
    
class JourneyLogEntry(Base):
    __tablename__ = "journey_log_entries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    chat_session_state_id = Column(Integer, ForeignKey("chatbot_session_states.id"), nullable=False, index=True) 
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    event_type = Column(String, nullable=False) 
    payload_json = Column(JSON, nullable=True) # Stores event-specific data

    session_state = relationship("ChatSessionState", back_populates="journey_log_entries")