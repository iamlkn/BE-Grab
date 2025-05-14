from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID
from pydantic import BaseModel
from sqlalchemy import select, update as sqlalchemy_update, delete as sqlalchemy_delete
from sqlalchemy.orm import Session # Import synchronous Session

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        stmt = select(self.model).where(self.model.id == id)
        # For sync, execute on session or use query
        return db.execute(stmt).scalar_one_or_none()
        # Alternative: return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100, **filters: Any
    ) -> List[ModelType]:
        stmt = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)
        stmt = stmt.offset(skip).limit(limit)
        return db.execute(stmt).scalars().all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = {}
        model_columns = {c.name for c in self.model.__table__.columns}
        for key, value in obj_in.dict().items():
            if key in model_columns:
                 obj_in_data[key] = value

        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        update_data = obj_in if isinstance(obj_in, dict) else obj_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update_atomic(
        self, db: Session, *, id: Any, values: Dict[str, Any]
    ) -> Optional[ModelType]:
        valid_values = {k: v for k, v in values.items() if hasattr(self.model, k)}
        if not valid_values:
             return self.get(db, id)

        # Synchronous update statement execution
        db.execute(
            sqlalchemy_update(self.model)
            .where(self.model.id == id)
            .values(**valid_values)
        )
        db.commit()
        # Need to re-fetch the object to get updated values in this simple sync version
        # Or explore options like session.expire(obj) before commit and then access
        return self.get(db, id) # Re-fetch after commit

    def remove(self, db: Session, *, id: Any) -> Optional[ModelType]:
        obj = self.get(db, id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj