from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


from sqlalchemy.orm import declarative_base

# Create async engine
engine = create_async_engine(
    settings.database_url,
    # This argument is specific to SQLite and required for FastAPI's async context
    connect_args={"check_same_thread": False},
    echo=settings.database_echo,
    future=True,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,  # Using bind=engine is slightly more explicit
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for your models to inherit from
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency to get a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
