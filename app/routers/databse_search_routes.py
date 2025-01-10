# app/routers/databse_search_routes.py
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from app.logic.databse_search_logic import DatabaseSearchLogic
from app.logic.multi_agent_workflow_logic import MultiAgentWorkflow

router = APIRouter(
    prefix="/database",
    tags=["database"],
)

database_search_logic = DatabaseSearchLogic()
multi_agent_workflow_logic = MultiAgentWorkflow()

@router.post("/ctc-search")
async def handle_ctc_search(search_request: dict = Body(...)):
    try:
        email = search_request.get("email")
        query = search_request.get("query")
        chat_id = search_request.get("chat_id")

        response = await database_search_logic.handle_ctc_search(query=query, user_email=email, chat_id=chat_id)
        # response = await multi_agent_workflow_logic.process_query(query=query, user_email=email, chat_id=chat_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
