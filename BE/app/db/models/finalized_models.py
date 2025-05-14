import uuid
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class FinalizedModel(Base):
    __tablename__ = "finalized_models"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("automl_sessions.id", ondelete="CASCADE"), # Points to AutoMLSession PK
        nullable=False,  # Cannot exist without a session
        #unique=True,     # Enforces one-to-one at DB level
        index=True       # Good to index foreign keys
    )
    model_name = Column(String, nullable=False)
    saved_model_path = Column(String, nullable=False)
    saved_metadata_path = Column(String, nullable=False)
    # model_uri_for_registry = Column(String, nullable=True) # URI for MLflow etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship back to session
    automl_sessions = relationship("AutoMLSession", back_populates="finalized_models")