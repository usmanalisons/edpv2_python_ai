# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()
# load_dotenv(dotenv_path="../../.env")

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PYTESSERACT_LOCAL_PATH = os.getenv("PYTESSERACT_LOCAL_PATH")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1536))
    CACHE_MAX_SIZE = int(100)
    COLLECTION_NAMES = {
       "policiesProcedures":  "policies_procedures_embeddings", 
        "oracleTrainings": "oracle_trainings_embeddings"
    }
    SQL_CONNECTION_STRINGS = {
        "intranet": os.getenv("SQL_CONNECTION_STRING_INTRANET"),
        "ctc": os.getenv("SQL_CONNECTION_STRING_CTC"),
        "quantum": os.getenv("SQL_CONNECTION_STRING_QUANTUM"),
    }
    BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")

settings = Settings()

# print(settings.SQL_CONNECTION_STRINGS)

# uvicorn app.main:app --reload --port 8001
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# pip install -r requirements.txt