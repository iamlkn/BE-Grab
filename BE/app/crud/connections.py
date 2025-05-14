from sqlalchemy.orm import Session
from app.db.models.connections import Connection

def get_connection_by_id(db: Session, conn_id: int) -> Connection | None:
    return db.query(Connection).filter(Connection.id == conn_id).first()

def get_connection(db: Session) -> list[Connection]:
    return db.query(Connection).all()
    
def create_connection(db: Session, type: str, host: str, port: int, database: str, username: str, password: str) -> Connection:
    c = Connection(type=type, host=host, port=port, database=database, username=username, password=password)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c