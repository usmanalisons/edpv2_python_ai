# app/routers/databse_search_routes.py,
from fastapi import APIRouter, HTTPException, Body, Request, Depends
from app.logic.databse_search_logic import DatabaseSearchLogic
# from app.logic.multi_agent_workflow_logic import MultiAgentWorkflow

router = APIRouter(
    prefix="/database",
    tags=["database"],
)

# multi_agent_workflow_logic = MultiAgentWorkflow()


async def get_database_search_logic(request: Request):
    chromadb_service = request.app.state.chromadb_service
    chat_memory_service = request.app.state.chat_memory_service
    return DatabaseSearchLogic(
        chroma_db_service=chromadb_service,
        chat_memory_service=chat_memory_service,
    )

@router.post("/ctc-search")
async def handle_ctc_search(search_request: dict = Body(...), logic: DatabaseSearchLogic = Depends(get_database_search_logic)):
    try:
        email = search_request.get("email")
        query = search_request.get("query")
        chat_id = search_request.get("chat_id")

        response = await logic.handle_ctc_search(query=query, user_email=email, chat_id=chat_id)
        # response = await multi_agent_workflow_logic.process_query(query=query, user_email=email, chat_id=chat_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))