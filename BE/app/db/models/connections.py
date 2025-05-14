from sqlalchemy import Column, Integer, String
from app.db.base import Base
from sqlalchemy.orm import relationship

class Connection(Base):
    __tablename__ = 'connections'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    type = Column(String, nullable=False) # postgres or ...
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    database = Column(String, nullable=False)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)
    
    datasets = relationship('Dataset', back_populates='connections', cascade='all, delete-orphan')