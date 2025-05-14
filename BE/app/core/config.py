from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    
    data_storage_dir: str = 'data'
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra='ignore'
        
settings = Settings()