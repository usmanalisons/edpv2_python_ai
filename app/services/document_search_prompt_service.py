# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai.chat_models import ChatOpenAI
# from app.utils.helper import Helper
# from langchain.schema import HumanMessage, AIMessage
# from typing import Tuple, List
# from langchain.memory import ConversationBufferMemory


# class DocumentSearchPromptService:
#     def __init__(self, model_name: str, api_key: str):
#         self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.7)

#     async def classify_question(self, question: str, memory: ConversationBufferMemory):
#         chat_history = memory.load_memory_variables({})
        
#         system_prompt = """Analyze the user question and chat history to determine which search system would be most appropriate.
#         Respond with exactly one of these options: POLICIES, ORACLE, CTC
        
#         Guidelines:
#         - POLICIES: For queries about company policies, procedures, guidelines, regulations, compliance, loan, social media, HR, departments etc
#         - ORACLE: For questions about Oracle systems, training materials, or Oracle-related procedures
#         - CTC (Cost to Complete): For queries about project costs, budgets, cost estimates, or financial data
        
#         Chat History:
#         {chat_history}
        
#         Current question: {question}
#         """
        
#         messages = [
#             {"role": "system", "content": system_prompt.format(
#                 chat_history=chat_history.get("chat_history", ""),
#                 question=question
#             )},
#             {"role": "user", "content": question}
#         ]

#         response = await self.llm.apredict_messages(messages)
#         return (response.content.strip().lower())

#     async def refine_user_question(self, question: str, previous_human_messagesList: List[str]) -> str:
#         numbered_messages = "\n".join(
#             [f"{i + 1} - {message}" for i, message in enumerate(previous_human_messagesList)]
#         )
        
#         merged_input = f"""
#         User's question: {question}
#         Conversation History:
#         {numbered_messages}
#         """

#         input_tokens = Helper.count_tokens(merged_input, model="gpt-4o")
#         max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
#         self.llm.max_tokens = min(max_completion_tokens, 16384)

#         prompt_template = PromptTemplate(
#             input_variables=["question", "numbered_messages"],
#             template="""
#             You are an intelligent AI assistant that refines user queries based on the provided conversation history.
#             Guidelines:
#             1. Analyze the user's current question and the conversation history below.
#             2. Prioritize the most recent messages (higher numbers) in the history to determine the context.
#             3. Dynamically decide whether the user's current question relates to the previous messages:
#             - If it is related, refine it by linking it to the most recent relevant message in the history.
#             - If it is unrelated, treat the current question as a standalone question.
#             4. Do not assume any specific patterns or keywords in the user's question. Instead, rely on the conversation's flow and context.
#             5. Refine the question into a single clear and precise question for use in a vector database search.
#             6. Do not make assumptions or add extra information beyond the input provided.

#             Numbered Conversation History (most recent):
#             {numbered_messages}

#             User's Current question:
#             {question}

#             Based on the above guidelines, provide:
#             - A single refined question for precise searching in the vector database.
#             - Return only the refined question without any extra text, quotes, or whitespace.
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         refined_question = await chain.arun({"question": question, "numbered_messages": numbered_messages})
#         return refined_question.strip()


#     async def generate_text_answer(self, question: str, context: str, memory, search_type: str = "Policies & Procedures") -> Tuple[str, bool]:
#         chat_history_string = self.summarize_messages(memory.chat_memory.messages) if memory.chat_memory.messages else ""

#         merged_input = f"""
#         Refined question: {question}
#         Search Type: {search_type}
#         Context:
#         {context}
#         Chat History:
#         {chat_history_string}
#         """

#         input_tokens = Helper.count_tokens(merged_input, model="gpt-4o")
#         max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
#         self.llm.max_tokens = min(max_completion_tokens, 16384)

#         prompt_template = PromptTemplate(
#             input_variables=["question", "search_type", "context", "chat_history"],
#             template="""
#                 You are an AI assistant engaging in a conversation with a user.
#                 Below is the user's question, the type of documents being searched, relevant context, and the chat history.

#                 question:
#                 {question}
#                 Search Type:
#                 {search_type}
#                 Context:
#                 {context}
#                 Chat History:
#                 {chat_history}

#                 Your Task:
#                 1. ONLY use the provided context, chat history, and search type to generate an answer, don't take anything from outside.
#                 2. If there is insufficient information, explicitly respond with: "Insufficient information in the context and chat history. Please provide more details."
#                 3. DO NOT guess or provide generalized answers. Strictly adhere to the provided input.
#                 4. Return the response in the following format:
#                 Answer: <your generated answer>
#                 AnswerBasedOnContext: <true/false>
#                 """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         response = await chain.arun({
#             "question": question,
#             "search_type": search_type,
#             "context": context,
#             "chat_history": chat_history_string
#         })

#         response_text = response.strip()

#         answer_start = response_text.find("Answer:") + len("Answer:")
#         context_start = response_text.find("AnswerBasedOnContext:")
#         answer = response_text[answer_start:context_start].strip()
#         answer_based_on_context = response_text[context_start + len("AnswerBasedOnContext:"):].strip()

#         answer_based_on_context = True if answer_based_on_context.lower() == "true" else False

#         return answer, answer_based_on_context


#     async def format_text_answer(self, answer: str, question: str, generate_title: bool) -> Tuple[str, str]:
#         generate_title_text = "True" if generate_title else "False"

#         merged_input = f"""
#         User's question: {question}
#         Generated Answer:
#         {answer}
#         Generate Title: {generate_title_text}
#         """

#         input_tokens = Helper.count_tokens(merged_input, model="gpt-4o")
#         max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
#         self.llm.max_tokens = min(max_completion_tokens, 16384)

#         prompt_template = PromptTemplate(
#             input_variables=["answer", "question", "generate_title"],
#             template="""
#             You are an AI assistant engaging in a conversation with a user.
#             The user's question is:
#             {question}
#             The generated answer is:
#             {answer}

#             Your task:
#             1. Format the answer as if you're responding naturally to the user, maintaining a conversational tone.
#             2. Ensure the answer is valid HTML but does not include <html>, <body>, or any header/footer tags or quotes and avoid h1 tag.
#             3. If 'generate_title' is True, create a concise and relevant title based on the question. Otherwise, return an empty string for the title.

#             Return the response in the following format:
#             Answer: <formatted conversational HTML answer>
#             Title: <title or empty string>
#             Remember strictly to not give any information from assumptions, only use context, chat history(memory), and question.
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         response = await chain.arun({
#             "answer": answer,
#             "question": question,
#             "generate_title": generate_title_text
#         })

#         response_text = response.strip()
#         answer_start = response_text.find("Answer:") + len("Answer:")
#         title_start = response_text.find("Title:")
#         formatted_answer = response_text[answer_start:title_start].strip()
#         title = response_text[title_start + len("Title:"):].strip()

#         return formatted_answer, title

#     @staticmethod
#     def calculate_available_tokens(total_limit, input_tokens):
#         return max(total_limit - input_tokens, 0)

#     @staticmethod
#     def summarize_messages(messages, max_length=1000, max_messages = 10):
#         if not messages:
#             return "No history available."
        
#         summary = ""
#         for msg in messages[-max_messages:]:
#             if isinstance(msg, HumanMessage):
#                 summary += f"Human: {msg.content[:max_length]}...\n"
#             elif isinstance(msg, AIMessage):
#                 summary += f"Assistant: {msg.content[:max_length]}...\n"
#         return summary.strip()


from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from typing import Dict, Optional
from app.services.token_service import TokenService

class DocumentSearchPromptService:
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0.7
        )
        self.token_service = TokenService(model_name=model_name)
        self.analyze_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an advanced AI assistant that analyzes user queries in two ways:
                1. Classification: Determine which system should handle the query
                2. Query Refinement: Either improve the query or request clarification
                
                Classification Guidelines:
                - POLICIES: Company policies, procedures, guidelines, regulations, compliance, loan, HR
                - ORACLE: Oracle systems, training materials, Oracle-related procedures
                - CTC: Project costs, budgets, cost estimates, financial data
                - EMPTY: When the query is unclear or needs clarification
                
                Refinement Guidelines:
                - For clear queries: Enhance them for better search results
                - For unclear queries: Create a user-friendly follow-up question that:
                  * Clearly explains what additional information is needed
                  * Uses natural, conversational language
                  * Provides examples when helpful
                  * Avoids technical jargon
                
                Format your response exactly as:
                CLASSIFICATION: <POLICIES/ORACLE/CTC/EMPTY>
                REFINED_QUERY: <refined query or follow-up question>"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Current Question: {question}
                Chat History:
                {chat_history}
                
                Analyze this query according to the guidelines."""
            )
        ])
        
        # Define the combined answer generation prompt template
        self.answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an AI assistant that generates comprehensive responses in two formats:
                1. A natural text response
                2. A formatted HTML response
                
                Guidelines:
                - Use ONLY the provided context and chat history
                - Maintain a professional yet conversational tone
                - Format numbers and data consistently
                - For HTML:
                  * Use proper HTML tags for structure
                  * Include appropriate formatting
                  * Avoid html/body/header tags
                  * Create clean, readable layouts
                  * Use tables for structured data
                
                Return your response in this exact format:
                TEXT_ANSWER: <natural language response>
                HTML_ANSWER: <formatted HTML version>
                CONTEXT_BASED: <true/false>
                TITLE: <title if new chat, empty string if not>"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Question: {question}
                Search Type: {search_type}
                Context: {context}
                Chat History: {chat_history}
                Generate Title: {generate_title}
                
                Generate both text and HTML responses."""
            )
        ])

    async def analyze_question(
        self,
        question: str,
        memory: ConversationBufferMemory
    ) -> Dict[str, str]:
       
        chat_history = self.token_service.summarize_chat_history(
            memory=memory,
            max_length=1000,
            max_messages=10
        )
        
        # Calculate available tokens for the response
        input_text = f"{question}\n{chat_history}"
        max_tokens = self.token_service.calculate_max_tokens(input_text)
        
        # Create and run the chain
        chain = LLMChain(
            llm=self.llm.with_config({"max_tokens": max_tokens}),
            prompt=self.analyze_prompt
        )
        
        response = await chain.arun({
            "question": question,
            "chat_history": chat_history
        })
        
        # Parse the response into components
        lines = response.strip().split('\n')
        result = {
            "classification": "",
            "refined_query": ""
        }
        
        for line in lines:
            if line.startswith("CLASSIFICATION:"):
                result["classification"] = line.replace("CLASSIFICATION:", "").strip()
            elif line.startswith("REFINED_QUERY:"):
                result["refined_query"] = line.replace("REFINED_QUERY:", "").strip()
        
        return result

    async def generate_answer(
        self,
        question: str,
        context: str,
        memory: ConversationBufferMemory,
        search_type: str = "Policies & Procedures",
        generate_title: bool = False
    ) -> Dict[str, str]:
       
        max_context_tokens = self.token_service.calculate_context_limit(
            query=question,
            chat_history=memory.chat_history
        )
        optimized_context = self.token_service.optimize_context(
            context=context,
            max_tokens=max_context_tokens
        )
        
        # Get chat history summary
        chat_history = self.token_service.summarize_chat_history(
            memory=memory
        )
        
        # Calculate available tokens for the response
        merged_input = f"""
        Question: {question}
        Context: {optimized_context}
        History: {chat_history}
        """
        max_tokens = self.token_service.calculate_max_tokens(merged_input)
        
        # Create and run the chain
        chain = LLMChain(
            llm=self.llm.with_config({"max_tokens": max_tokens}),
            prompt=self.answer_prompt
        )
        
        response = await chain.arun({
            "question": question,
            "search_type": search_type,
            "context": optimized_context,
            "chat_history": chat_history,
            "generate_title": str(generate_title)
        })
        
        # Parse the response into components
        lines = response.strip().split('\n')
        result = {
            "text_answer": "",
            "html_answer": "",
            "context_based": False,
            "title": ""
        }
        
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith("TEXT_ANSWER:"):
                if current_section:
                    result[current_section.lower()] = '\n'.join(current_content).strip()
                current_section = "text_answer"
                current_content = [line.replace("TEXT_ANSWER:", "").strip()]
            elif line.startswith("HTML_ANSWER:"):
                if current_section:
                    result[current_section.lower()] = '\n'.join(current_content).strip()
                current_section = "html_answer"
                current_content = [line.replace("HTML_ANSWER:", "").strip()]
            elif line.startswith("CONTEXT_BASED:"):
                if current_section:
                    result[current_section.lower()] = '\n'.join(current_content).strip()
                result["context_based"] = line.replace("CONTEXT_BASED:", "").strip().lower() == "true"
            elif line.startswith("TITLE:"):
                if current_section:
                    result[current_section.lower()] = '\n'.join(current_content).strip()
                result["title"] = line.replace("TITLE:", "").strip()
            else:
                if current_section:
                    current_content.append(line)
        
        if current_section:
            result[current_section.lower()] = '\n'.join(current_content).strip()
        
        return result