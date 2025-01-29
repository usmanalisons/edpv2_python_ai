from typing import List, Dict
from langchain.schema import HumanMessage, AIMessage
from app.utils.helper import Helper

class TokenService:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.max_input_tokens = 128000  # GPT-4o max input tokens
        self.max_output_tokens = 16384  # GPT-4o max output tokens
        self.safety_margin = 1000  # Buffer for system messages and safety

    def calculate_max_tokens(self, input_text: str) -> int:
        """Calculate maximum available tokens for completion"""
        input_tokens = Helper.count_tokens(input_text, self.model_name)
        available_tokens = self.max_input_tokens - input_tokens - self.safety_margin
        return min(available_tokens, self.max_output_tokens)

    def optimize_context(self, context: str, max_tokens: int) -> str:
        """Optimize context to fit within token limits"""
        context_tokens = Helper.count_tokens(context, self.model_name)
        
        if context_tokens <= max_tokens:
            return context

        # Split context into paragraphs and reconstruct within token limit
        paragraphs = context.split('\n\n')
        optimized_paragraphs = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = Helper.count_tokens(paragraph, self.model_name)
            if current_tokens + paragraph_tokens <= max_tokens:
                optimized_paragraphs.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                break

        return '\n\n'.join(optimized_paragraphs)

    def summarize_chat_history(
        self,
        memory,
        max_length: int = 1000,
        max_messages: int = 10
    ) -> str:
        """Summarize chat history within token limits"""
        if not memory.chat_memory.messages:
            return "No history available."
        
        summary = []
        current_tokens = 0
        max_tokens = self.calculate_max_tokens("") // 4  # Use 1/4 of available tokens for history
        
        for msg in reversed(memory.chat_memory.messages[-max_messages:]):
            content = msg.content[:max_length]
            msg_tokens = Helper.count_tokens(content, self.model_name)
            
            if current_tokens + msg_tokens > max_tokens:
                break
                
            if isinstance(msg, HumanMessage):
                summary.insert(0, f"Human: {content}")
            elif isinstance(msg, AIMessage):
                summary.insert(0, f"Assistant: {content}")
                
            current_tokens += msg_tokens
        
        return "\n".join(summary)

    def optimize_documents(
        self,
        documents: List[Dict],
        max_combined_tokens: int
    ) -> List[Dict]:
        """Optimize document content to fit within token limits"""
        optimized_docs = []
        current_tokens = 0

        for doc in documents:
            doc_content = doc.get('page_content', '')
            doc_tokens = Helper.count_tokens(doc_content, self.model_name)
            
            # If single document exceeds half of max tokens, truncate it
            if doc_tokens > max_combined_tokens // 2:
                doc_content = self.optimize_context(doc_content, max_combined_tokens // 2)
                doc_tokens = Helper.count_tokens(doc_content, self.model_name)
            
            if current_tokens + doc_tokens <= max_combined_tokens:
                doc['page_content'] = doc_content
                optimized_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break

        return optimized_docs

    def calculate_context_limit(
        self,
        query: str,
        chat_history: str
    ) -> int:
        """Calculate maximum tokens available for context"""
        query_tokens = Helper.count_tokens(query, self.model_name)
        history_tokens = Helper.count_tokens(chat_history, self.model_name)
        
        available_tokens = (
            self.max_input_tokens -
            query_tokens -
            history_tokens -
            self.safety_margin
        )
        
        return max(0, available_tokens)