"""Shared SQLAlchemy database configuration and dependency helpers."""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Declarative base shared by all SQLAlchemy models."""


engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for FastAPI dependency injection."""

    async with SessionLocal() as session:
        yield session


async def verify_database_connection() -> None:
    """Fail fast if PostgreSQL is unreachable or the URL is invalid."""

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))