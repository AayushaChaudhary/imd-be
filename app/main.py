import time
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.lifespan import lifespan
from app.api.v1.endpoints import router as api_router
from app.utils.logger import setup_logging

setup_logging()
logger = structlog.get_logger()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered music recommendation system",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=round(process_time, 4),
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Validation error", errors=exc.errors(), url=str(request.url))
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "status_code": 422,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception - THIS IS LIKELY CAUSING THE 502 ERROR",
        error=str(exc),
        url=str(request.url),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error. Check server logs.",
            "status_code": 500,
        },
    )


app.include_router(api_router, prefix="/api/v1", tags=["v1"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name} v{settings.version}",
        "docs": app.docs_url,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": settings.version}
