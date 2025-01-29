import os
import logging
from uuid import uuid4
from fastapi import UploadFile
from app.utils.file_loader import load_text_from_file
from app.services.text_processor_service import TextProcessorService
from app.services.document_search_prompt_service import DocumentSearchPromptService
from app.utils.helper import Helper
from app.core.config import settings
from app.core.constants import MAX_TOKENS_PER_REQUEST
from app.services.sql_db_service import SQLDatabaseService
from app.services.chat_memory_service import ChatMemoryService
from app.services.chroma_db_service import ChromaDBService
from app.models.request_models import SendChatRequestUser


class ChatLogic:
    def __init__(self, chroma_db_service, chat_memory_service):
        self.chroma_db_service: ChromaDBService = chroma_db_service
        self.chat_memory_service: ChatMemoryService = chat_memory_service
        self.intranet_db_service = SQLDatabaseService(connection_name="intranet")
        self.quantum_db_service = SQLDatabaseService(connection_name="quantum")
        self.text_processor_service = TextProcessorService(model_name=settings.OPENAI_EMBEDDING_MODEL)
        self.prompt_service = DocumentSearchPromptService(model_name="gpt-4o", api_key=settings.OPENAI_API_KEY)

    def get_employee_chats(self, email, type):
        chats = self.intranet_db_service.get_employee_chats(email, type)
        return {
            "data": chats
        }
    
    def get_chat_details(self, chat_id):
        chat = self.intranet_db_service.get_chat_by_id(chat_id)
        if chat:
            messages = self.intranet_db_service.get_chat_messages(chat_id)
            chat['messages'] = messages
        return {
            "data": chat
        }
    
    def get_chat_messages(self, chat_id):
        messages = self.intranet_db_service.get_chat_messages(chat_id)
        return {
            "data": messages
        }


    async def send_chat_message(self, question: str, user: SendChatRequestUser, chat_id: str = None,):
        collection_key = "policiesProcedures"
        chat_memory = None
        is_new_chat = False

        chat = None
        if not chat_id:
            is_new_chat = True
            chat_id = str(uuid4())
            chat_memory = await self.chat_memory_service.create_memory(chat_id, [])
        else:
            chat = self.intranet_db_service.get_chat_by_id(chat_id)
            if not chat:
                return {"message": "Chat ID not found."}

            if await self.chat_memory_service.has_memory(chat_id):
                chat_memory = await self.chat_memory_service.get_memory(chat_id)
            else:
                previous_messages = self.intranet_db_service.get_chat_messages(chat_id)
                chat_memory = await self.chat_memory_service.create_memory(chat_id, previous_messages)

        response = await self.prompt_service.analyze_question(question, chat_memory)

        # Ensure the response is a serializable object (e.g., a dictionary)
        return {
            "data": {
                "type": "chat_memory",  # Replace with actual data you want to return
                "chat_id": chat_id,
                "is_new_chat": is_new_chat,
                "is_new_chat": response,
                # "chat_memory": chat_memory,  # Ensure this is serializable
            }
        }