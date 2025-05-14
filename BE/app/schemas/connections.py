from pydantic import BaseModel

class ConnectionId(BaseModel):
    id: int
    
    class Config:
        from_attributes = True

class ConnectionOut(BaseModel):
    id: int
    type: str
    host: str
    port: int
    database: str
    username: str
    
    class Config:
        from_attributes = True
        
class ConnectionCreate(BaseModel):
    type: str
    host: str
    port: int
    database: str
    username: str
    password: str