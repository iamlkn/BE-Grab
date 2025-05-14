from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_db
from app.schemas.connections import ConnectionOut, ConnectionCreate, ConnectionId
from app.crud.connections import get_connection, create_connection


router = APIRouter(
    prefix='/v1',
    tags=['connections'],
)

# Get all existing connection
@router.get('/connections/', response_model=list[ConnectionOut])
async def get_connections(db: Session = Depends(get_db)):
    return get_connection(db)

# Add Connections
@router.post('/connections/')
async def add_connection(payload: ConnectionCreate, db: Session = Depends(get_db)):
    c = create_connection(
        db,
        type=payload.type,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        username=payload.username,
        password=payload.password
    )
    
    return ConnectionId(id=c.id)

