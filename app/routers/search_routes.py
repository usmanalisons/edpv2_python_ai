from fastapi import APIRouter, Depends, UploadFile, Request
from app.logic.document_search_logic import DocumentSearchLogic

router = APIRouter(
    prefix="/document",
    tags=["document"],
)

async def get_document_search_logic(request: Request):
    chromadb_service = request.app.state.chromadb_service
    chat_memory_service = request.app.state.chat_memory_service

    return DocumentSearchLogic(
        chroma_db_service=chromadb_service,
        chat_memory_service=chat_memory_service,
    )

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile, logic: DocumentSearchLogic = Depends(get_document_search_logic)):
    return await logic.handle_file_upload(file)

@router.get("/process-policies-procedures")
async def process_policies_procedures(logic: DocumentSearchLogic = Depends(get_document_search_logic)):
    return await logic.process_policies_procedures_from_db()

@router.post("/search-policies-procedures")
async def search_policies_procedures(
    search_request: dict, 
    logic: DocumentSearchLogic = Depends(get_document_search_logic)
):
    email = search_request.get("email")
    query = search_request.get("query")
    department_code = search_request.get("department_code")
    company_code = search_request.get("company_code")
    chat_id = search_request.get("chat_id")

    return await logic.handle_policies_procedure_search(email, department_code, company_code, query, chat_id)

@router.get("/process-oracle-trainings")
async def process_oracle_trainings(logic: DocumentSearchLogic = Depends(get_document_search_logic)):
    return await logic.process_oracle_trainings_from_db()

@router.post("/search-oracle-trainings")
async def search_oracle_trainings(
    search_request: dict, 
    logic: DocumentSearchLogic = Depends(get_document_search_logic)
):
    email = search_request.get("email")
    query = search_request.get("query")
    chat_id = search_request.get("chat_id")

    return await logic.handle_oracle_trainings_search(email, query, chat_id)
