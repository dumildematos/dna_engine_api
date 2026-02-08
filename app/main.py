from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastapi.staticfiles import StaticFiles
from .brain import brain
from .api_v1 import router as api_router
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models into GPU
    brain.load_models()
    yield
    # Shutdown: Clean up if necessary
    print("Shutting down Brain...")

app = FastAPI(lifespan=lifespan, debug=True)

origins = [
    "http://localhost:4200",  # Angular default
    "http://127.0.0.1:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, GET, etc.
    allow_headers=["*"],  # Allows Content-Type, Authorization, etc.
)

# Include our fashion routes
app.include_router(api_router, prefix="/api/v1")
# app.mount("/images", StaticFiles(directory="data/raw"), name="images")
@app.get("/")
async def root():
    return {"message": "DNA Engine is running"}