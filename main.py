from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import router as api_v1_router
from app.config import ALLOWED_ORIGINS, APP_VERSION, logger
from app.middlewares.auth import JWTAuthMiddleware
from app.services.liveness_service import warmup_liveness_model
from app.services.redis_service import close_redis, init_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the lightweight liveness model and Redis client on boot.
    """
    logger.info(f"🚀 Booting Liveness Detection Engine v{APP_VERSION} ...")
    warmup_liveness_model()
    await init_redis()
    yield
    await close_redis()
    logger.info("🔌 Shutting down Liveness Engine...")


app = FastAPI(
    title="Liveness Detection Engine",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Authentication middleware (runs on all non-exempt routes)
app.add_middleware(JWTAuthMiddleware)

# CORS middleware (securely parameterized from environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full trace server-side only — never expose internals to clients (fixes python:S8572)
    logger.exception("Unhandled server exception occurred")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Terjadi kesalahan internal pada server."},
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint to check if the API is online and verify deployed version."""
    return {
        "message": "Welcome to Liveness Detection Engine API",
        "status": "running",
        "version": APP_VERSION,
    }


# Include Modular API Routers
app.include_router(api_v1_router)


if __name__ == "__main__":
    import uvicorn

    # Bind to 127.0.0.1 for secure local execution (fixes SonarCloud BLOCKER python:S8392)
    logger.info(f"🚀 Starting local development server v{APP_VERSION} on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
