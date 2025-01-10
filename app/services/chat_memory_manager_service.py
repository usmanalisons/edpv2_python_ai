import asyncio
from typing import Optional, Dict, List
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms import OpenAI
from langchain.schema import HumanMessage, AIMessage

class ChatMemoryManager:
    def __init__(self, openai_api_key: str, max_token_limit: int = 1000, max_messages: int = 10):
        self.openai_api_key = openai_api_key
        self.max_token_limit = max_token_limit
        self.max_messages = max_messages
        self.memory_sessions: Dict[str, ConversationSummaryBufferMemory] = {}
        self.lock = asyncio.Lock()  # Use asyncio.Lock for async compatibility

    async def has_memory(self, chat_id: str) -> bool:
        async with self.lock:  # Use async context manager
            return chat_id in self.memory_sessions

    async def get_memory(self, chat_id: str) -> Optional[ConversationSummaryBufferMemory]:
        async with self.lock:  # Use async context manager
            return self.memory_sessions.get(chat_id)

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

        async with self.lock:  # Use async context manager
            self.memory_sessions[chat_id] = memory
        return memory

    async def clear_memory(self, chat_id: str):
        async with self.lock:  # Use async context manager
            if chat_id in self.memory_sessions:
                del self.memory_sessions[chat_id]

    async def add_user_message(self, chat_id: str, text: str):
        async with self.lock:  # Use async context manager
            if chat_id not in self.memory_sessions:
                raise ValueError(f"No memory found for chat_id: {chat_id}")
            self.memory_sessions[chat_id].chat_memory.add_message(HumanMessage(content=text))

    async def add_assistant_message(self, chat_id: str, text: str):
        async with self.lock:  # Use async context manager
            if chat_id not in self.memory_sessions:
                raise ValueError(f"No memory found for chat_id: {chat_id}")
            self.memory_sessions[chat_id].chat_memory.add_message(AIMessage(content=text))
