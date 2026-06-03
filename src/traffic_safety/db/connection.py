"""SQLAlchemy engine, session factory, and database URL resolution."""

from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from traffic_safety.config import settings

load_dotenv()


def get_database_url() -> str:
    """
    Resolve the PostgreSQL connection URL.

    Priority:
        1. DATABASE_URL from environment or .env
        2. Assembled URL from POSTGRES_* settings
    """
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url
    return settings.database_url


engine = create_engine(get_database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_session() -> Generator[Session, None, None]:
    """
    Yield a database session and ensure it is closed afterward.

    Usage:
        with contextlib.closing(next(get_session())) as session: ...
        # or in FastAPI / dependency injection frameworks that consume generators
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
