from fastapi import APIRouter, Depends, Request
from app.logic.chat_logic import ChatLogic
from app.models.request_models import SendChatRequest, SendChatRequestUser

router = APIRouter(
    prefix="/chats",
    tags=["chats"],
)

async def get_chat_logic(request: Request):
    chromadb_service = request.app.state.chromadb_service
    chat_memory_service = request.app.state.chat_memory_service

    return ChatLogic(
        chroma_db_service=chromadb_service,
        chat_memory_service=chat_memory_service,
    )

@router.get("")
async def get_employee_chats(
    email: str, 
    type: str, 
    logic: ChatLogic = Depends(get_chat_logic)
):
    return logic.get_employee_chats(email, type)

@router.get("/{id}")
async def get_chat_details(
    id: str,
    logic: ChatLogic = Depends(get_chat_logic)
):
    return logic.get_chat_details(id)

@router.get("/messages/{id}")
async def get_chat_messages(
    id: str,
    logic: ChatLogic = Depends(get_chat_logic)
):
    return logic.get_chat_messages(id)

@router.post("/send")
async def send_chat_message(
    request: SendChatRequest,
    logic: ChatLogic = Depends(get_chat_logic)
):
    user = request.user
    question = request.question
    chat_id = request.chat_id

    return await logic.send_chat_message(chat_id=chat_id, user=user, question=question)