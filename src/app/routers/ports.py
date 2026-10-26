from typing import Annotated

from fastapi import APIRouter, Depends

from app.db import Database
from app.deps import get_db
from app.schemas import Port

router = APIRouter(prefix="/ports", tags=["ports"])


@router.get("", response_model=list[Port])
def list_ports(db: Annotated[Database, Depends(get_db)]) -> list[Port]:
    """Reference list of ports: the assistant uses it to turn names into codes."""
    return db.query("SELECT * FROM ports ORDER BY port_code")
