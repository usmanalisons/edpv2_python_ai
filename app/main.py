from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import document_search_routes, databse_search_routes, auth_routes
from app.dependencies import get_chromedb_service, get_chat_memory_manager, get_sql_database_service
import logging
from contextlib import asynccontextmanager
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting application...")

    chromadb_service = get_chromedb_service()
    chat_memory_manager = get_chat_memory_manager()
    intranet_db_service = get_sql_database_service("intranet")
    quantum_db_service = get_sql_database_service("quantum")

    app.state.chromadb_service = chromadb_service
    app.state.chat_memory_manager = chat_memory_manager
    app.state.intranet_db_service = intranet_db_service
    app.state.quantum_db_service = quantum_db_service

    try:
        with intranet_db_service.get_session() as session:
            session.execute(text("SELECT 1"))
        with quantum_db_service.get_session() as session:
            session.execute(text("SELECT 1"))
        logging.info("Dependencies initialized successfully.")
        yield
    finally:
        logging.info("Shutting down application...")
        logging.info("Cleanup completed.")

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
app.include_router(auth_routes.router, prefix="/api")

@app.get("/api")
async def root():
    return {"message": "Home Route"}

app.mount("/api", app)
