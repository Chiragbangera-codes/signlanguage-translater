import logging
from contextlib import asynccontextmanager

import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Load env variables
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backend_api")

from .api.v1.predict import router as predict_router
from .services.model_loader import ModelLoaderService

model_loader = ModelLoaderService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager that handles startup initialization of model assets."""
    logger.info("Starting SignSpeak AI FastAPI Server...")
    try:
        model_loader.initialize()
    except Exception as e:
        logger.error(f"Failed to load ML model checkpoints: {e}")
        # Server can start even if model loading fails,
        # but prediction requests will return 503 Service Unavailable.
    yield
    logger.info("Shutting down SignSpeak AI FastAPI Server...")

app = FastAPI(
    title="SignSpeak AI API",
    description="Real-time Indian Sign Language (ISL) Translation Inference API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configurations
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire up routers
app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])

@app.get("/api/v1/health", summary="Health Check Status")
async def health_check():
    """Returns the API service health status, model load status, and API version."""
    model_loaded = model_loader.is_loaded
    # Dynamically extract TensorFlow version major/minor (e.g. 2.16.1 -> 2.x)
    tf_version_parts = tf.__version__.split('.')
    tf_version_major = f"{tf_version_parts[0]}.x" if tf_version_parts else "2.x"
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "tensorflow": tf_version_major
    }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Overrides default FastAPI validation handler to return structured error payload."""
    errors = exc.errors()
    message = "Invalid input payload."
    if errors:
        first_err = errors[0]
        # Clean location path (e.g. body -> left_hand -> left_hand)
        loc = " -> ".join(str(part) for part in first_err.get("loc", []))

        msg = first_err.get("msg", "Validation failed")
        loc_clean = loc.replace("body -> ", "")
        message = f"Validation failed at '{loc_clean}': {msg}"

    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_INPUT",
            "message": message
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Intercepts Starlette HTTPExceptions to return formatted error JSON."""
    error_code = "SERVER_ERROR"
    if exc.status_code == 503:
        error_code = "SERVICE_UNAVAILABLE"
    elif exc.status_code == 400:
        error_code = "INVALID_INPUT"
    elif exc.status_code == 404:
        error_code = "NOT_FOUND"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_code,
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to capture unhandled errors and return structured JSON."""
    logger.error(f"Unhandled server error on request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "SERVER_ERROR",
            "message": "Internal server error. Please check system logs."
        }
    )


@app.get("/", summary="API Root Portal")
async def root():
    return {
        "message": "Welcome to SignSpeak AI Translation API.",
        "health_check": "/api/v1/health",
        "docs": "/docs"
    }
