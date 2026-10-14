from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import logger, APP_VERSION
from app.services.liveness_service import warmup_liveness_model
from app.api.v1.endpoints import router as api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the lightweight liveness model into RAM on server boot.
    """
    logger.info(f"🚀 Booting Liveness Detection Engine v{APP_VERSION} ...")
    warmup_liveness_model()
    yield
    logger.info("🔌 Shutting down Liveness Engine...")


app = FastAPI(
    title="Liveness Detection Engine",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
