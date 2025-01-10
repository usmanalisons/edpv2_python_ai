import logging
from app.services.chroma_db_service import ChromaDBService
from app.services.chat_memory_manager_service import ChatMemoryManager
from app.services.sql_db_service import SQLDatabaseService
from app.core.config import settings

PERSIST_DIRECTORY = "./chromadb_data"
MODEL_NAME = "text-embedding-ada-002"

def get_chromedb_service():
    return ChromaDBService(
        persist_directory=PERSIST_DIRECTORY,
        collection_names=settings.COLLECTION_NAMES,
        model_name=MODEL_NAME,
    )

def get_chat_memory_manager():
    return ChatMemoryManager(openai_api_key=settings.OPENAI_API_KEY)

def get_sql_database_service(connection_name: str = "intranet"):
    return SQLDatabaseService(connection_name=connection_name)
