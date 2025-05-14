from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class CleaningJob(Base):
    __tablename__ = 'cleaning_jobs'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    status = Column(String, default="pending", nullable=False) # 'pending', 'running', 'completed', 'failed' 
    config = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    datasets = relationship("Dataset", back_populates='cleaning_jobs')