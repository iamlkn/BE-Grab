from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core import config as settings  # Import settings (database URL)

engine = create_engine(
    settings.database_url,  # Point to PostgreSQL database
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
