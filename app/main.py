from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import document_search_routes, databse_search_routes, auth_routes, chat_routes
from app.dependencies import get_chromedb_service, get_chat_memory_service
import logging
from contextlib import asynccontextmanager
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting application...")

    chromadb_service = get_chromedb_service()
    chat_memory_service = get_chat_memory_service()

    app.state.chromadb_service = chromadb_service
    app.state.chat_memory_service = chat_memory_service

    task = asyncio.create_task(chat_memory_service.clear_inactive_sessions())

    try:
        yield
    finally:
        task.cancel()
        logging.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/charts", StaticFiles(directory="app/charts"), name="charts")

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={"message": "The requested resource was not found"}
    )

app.include_router(document_search_routes.router, prefix="/api")
app.include_router(databse_search_routes.router, prefix="/api")
app.include_router(chat_routes.router, prefix="/api")
app.include_router(auth_routes.router, prefix="/api")

@app.get("/api")
async def root():
    return {"message": "Home Route"}

app.mount("/api", app)
