import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms import OpenAI
from langchain.schema import HumanMessage, AIMessage

class ChatMemoryService:
    def __init__(
        self, 
        openai_api_key: str, 
        max_token_limit: int = 1000, 
        max_messages: int = 10, 
        inactivity_timeout: int = 10
    ):
        self.openai_api_key = openai_api_key
        self.max_token_limit = max_token_limit
        self.max_messages = max_messages
        self.inactivity_timeout = inactivity_timeout  # Timeout in minutes
        self.memory_sessions: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()

    async def has_memory(self, chat_id: str) -> bool:
        async with self.lock:
            return chat_id in self.memory_sessions

    async def get_memory(self, chat_id: str) -> Optional[ConversationSummaryBufferMemory]:
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                session["last_interaction"] = datetime.now()
                return session["memory"]
            return None

    async def create_memory(self, chat_id: str, chat_messages: List[Dict[str, str]] = None):
        memory = ConversationSummaryBufferMemory(
            llm=OpenAI(
                temperature=0,
                openai_api_key=self.openai_api_key
            ),
            memory_key='chat_history',
            max_token_limit=self.max_token_limit,
            return_messages=True
        )
        if chat_messages:
            for msg in chat_messages:
                if msg["message_type"] == "HUMAN":
                    memory.chat_memory.add_message(HumanMessage(content=msg["message_text"]))
                elif msg["message_type"] == "ASSISTANT":
                    memory.chat_memory.add_message(AIMessage(content=msg["message_text"]))

        async with self.lock:
            self.memory_sessions[chat_id] = {"memory": memory, "last_interaction": datetime.now()}
        return memory

    async def clear_memory(self, chat_id: str):
        async with self.lock:
            if chat_id in self.memory_sessions:
                del self.memory_sessions[chat_id]

    async def add_human_message(self, chat_id: str, text: str):
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                session["memory"].chat_memory.add_message(HumanMessage(content=text))
                session["last_interaction"] = datetime.now()

    async def add_assistant_message(self, chat_id: str, text: str):
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                session["memory"].chat_memory.add_message(AIMessage(content=text))
                session["last_interaction"] = datetime.now()

    async def get_human_messages(self, chat_id: str) -> List[str]:
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                return [
                    msg.content for msg in session["memory"].chat_memory.messages
                    if isinstance(msg, HumanMessage)
                ]
            return []

    async def get_assistant_messages(self, chat_id: str) -> List[str]:
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                return [
                    msg.content for msg in session["memory"].chat_memory.messages
                    if isinstance(msg, AIMessage)
                ]
            return []

    async def get_all_messages(self, chat_id: str) -> List[Dict[str, str]]:
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                return [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content
                    }
                    for msg in session["memory"].chat_memory.messages
                ]
            return []

    async def get_all_memories(self) -> Dict[str, List[Dict[str, str]]]:
        async with self.lock:
            all_memories = {}
            for chat_id, session in self.memory_sessions.items():
                messages = [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content
                    }
                    for msg in session["memory"].chat_memory.messages
                ]
                all_memories[chat_id] = messages
            return all_memories

    async def get_chat_summary(self, chat_id: str) -> str:
        async with self.lock:
            session = self.memory_sessions.get(chat_id)
            if session:
                return session["memory"].moving_summary_buffer
            return ""

    async def clear_inactive_sessions(self):
        while True:
            async with self.lock:
                now = datetime.now()
                inactive_chats = [
                    chat_id for chat_id, session in self.memory_sessions.items()
                    if now - session["last_interaction"] > timedelta(minutes=self.inactivity_timeout)
                ]
                for chat_id in inactive_chats:
                    del self.memory_sessions[chat_id]
            await asyncio.sleep(60)
