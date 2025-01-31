# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage, AIMessage, SystemMessage
# from langchain.memory import ConversationBufferMemory
# from typing import Dict, Optional
# from app.services.token_service import TokenService
# from typing import Dict, List, Optional
# from dataclasses import dataclass, field

# @dataclass
# class AnalysisResult:
#     classification: str
#     refined_question: str
#     error: str = ""
#     chart_types: List[str] = field(default_factory=list)

# class ChatPromptService:
#     def __init__(self, model_name: str, api_key: str):
#         self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.7)
#         self.token_service = TokenService(model_name=model_name)
        
#         self.analysis_prompt = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(
#                 """You are an advanced AI assistant that analyzes user queries in multiple ways:
#                 1. Classification: Determine which system should handle the question
#                 2. question Refinement: Either improve the question or request clarification
#                 3. Error Detection: Identify if the question is unclear or incomplete
#                 4. Visualization Needs: Determine if the question requires any charts or visualizations

#                 **Classification Guidelines:**
#                 - POLICIES: Company policies, procedures, guidelines, regulations, compliance
#                 - ORACLE: Oracle systems, training materials, Oracle-related procedures
#                 - CTC: Project costs, budgets, cost estimates, financial data
#                 - EMPTY: When the question is unclear or needs clarification

#                 **Refinement Guidelines:**
#                 - For clear queries: Enhance them for better search results
#                 - For unclear queries: Create a user-friendly follow-up question that:
#                   * Clearly explains what additional information is needed
#                   * Uses natural, conversational language
#                   * Provides examples when helpful
#                   * Avoids technical jargon

#                 **Visualization Guidelines:**
#                 - Identify if the question requires any of the following chart types:
#                   * bar_chart
#                   * line_chart
#                   * pie_chart
#                   * column_chart

#                 **Error Handling:**
#                 - If the question is unclear or incomplete, provide an error message explaining the issue.

#                 **Database Context:**
#                 - Here is some sample CTC data from the database to help you understand the terms and structure:
#                 {sample_data}

#                 **Response Format:**
#                 CLASSIFICATION: <POLICIES/ORACLE/CTC/EMPTY>
#                 REFINED_QUESTION: <refined question or follow-up question>
#                 ERROR: <error message if unclear, empty otherwise>
#                 CHART_TYPES: <comma-separated list of requested chart types if any>"""
#             ),
#             HumanMessagePromptTemplate.from_template(
#                 """Database Schema:
#                 {schema_info}

#                 Chat History:
#                 {chat_history}

#                 User Question:
#                 {question}

#                 Analyze this question according to the all guidelines."""
#             )
#         ])


#     # async def analyze_question(
#     #     self,
#     #     question: str,
#     #     memory: ConversationBufferMemory,
#     #     schema_info: Optional[str] = None,
#     #     sample_data: Optional[List[Dict]] = None
#     # ) -> AnalysisResult:
#     #     chat_history = self._get_chat_history(memory)
        
#     #     chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
#     #     response = await chain.arun({
#     #         "question": question,
#     #         "chat_history": chat_history,
#     #         "schema_info": schema_info or "",
#     #         "sample_data": self._format_sample_data(sample_data) if sample_data else ""
#     #     })
        
#     #     return self._parse_response(response)

#     async def analyze_question(
#         self,
#         question: str,
#         memory: ConversationBufferMemory,
#         schema_info: Optional[str] = None,
#         sample_data: Optional[List[Dict]] = None
#     ) -> AnalysisResult:
#         # Get summarized chat history within token limits
#         chat_history = self.token_service.summarize_chat_history(memory)
        
#         # Calculate available tokens for context
#         max_context_tokens = self.token_service.calculate_context_limit(
#             question, chat_history
#         )
        
#         # Optimize schema info and sample data to fit within context limit
#         optimized_schema_info = self.token_service.optimize_context(
#             schema_info or "", max_context_tokens // 2
#         )
#         optimized_sample_data = self._format_sample_data(
#             self.token_service.optimize_documents(sample_data or [], max_context_tokens // 2)
#         )
        
#         chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
#         response = await chain.arun({
#             "question": question,
#             "chat_history": chat_history,
#             "schema_info": optimized_schema_info,
#             "sample_data": optimized_sample_data
#         })
        
#         return self._parse_response(response)

#     def _get_chat_history(self, memory: ConversationBufferMemory, max_messages: int = 3) -> str:
#         if not memory or not memory.chat_memory.messages:
#             return ""
        
#         recent_messages = []
#         for msg in list(reversed(memory.chat_memory.messages))[:max_messages]:
#             prefix = "Human: " if isinstance(msg, HumanMessage) else "Assistant: "
#             content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
#             recent_messages.insert(0, f"{prefix}{content}")
            
#         return "\n".join(recent_messages)

#     @staticmethod
#     def _format_sample_data(data: List[Dict], max_rows: int = 2) -> str:
#         if not data:
#             return ""
        
#         formatted_rows = []
#         for i, row in enumerate(data[:max_rows], 1):
#             items = [f"{k}: {v:,}" if isinstance(v, (int, float)) else f"{k}: {v}"
#                     for k, v in row.items()]
#             formatted_rows.append(f"Row {i}: " + ", ".join(items))
        
#         return "\n".join(formatted_rows)

#     @staticmethod
#     def _parse_response(response: str) -> AnalysisResult:
#         result = AnalysisResult(
#             classification="",
#             refined_question="",
#             error="",
#             chart_types=[]
#         )
        
#         for line in response.strip().split('\n'):
#             if line.startswith('CLASSIFICATION:'):
#                 result.classification = line.replace('CLASSIFICATION:', '').strip().upper()
#             elif line.startswith('REFINED_QUESTION:'):
#                 result.refined_question = line.replace('REFINED_QUESTION:', '').strip()
#             elif line.startswith('ERROR:'):
#                 result.error = line.replace('ERROR:', '').strip()
#             elif line.startswith('CHART_TYPES:'):
#                 chart_types = line.replace('CHART_TYPES:', '').strip()
#                 if chart_types.lower() not in ['none', 'n/a', '']:
#                     result.chart_types = [t.strip() for t in chart_types.split(',') if t.strip()]
        
#         return result

    


from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from app.services.token_service import TokenService

@dataclass
class AnalysisResult:
    classification: str
    refined_question: str
    error: str = ""
    chart_types: List[str] = field(default_factory=list)

class ChatPromptService:
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.7)
        self.token_service = TokenService(model_name=model_name)
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an advanced AI assistant that classifies and refines user questions. Your job is to:
                1️⃣ **Classify the question** → Determine which system should handle it.
                2️⃣ **Refine the question** → If unclear, improve it or request clarification.
                3️⃣ **Detect errors** → If missing details, guide the user to clarify.
                4️⃣ *Identify chart types** → Specify the type of chart needed, if any (e.g., bar_chart, line_chart).

                ## 🔹 **Classification Rules**
                - **POLICIES** → Questions about company policies, procedures, compliance, HR, IT, finance, or security.
                - **ORACLE** → Oracle systems, training materials, and Oracle-related procedures.
                - **CTC** → Cost tracking, project budgets, financial estimates.
                - **NONE** → Only if the question is ambiguous or unrelated to company topics.

                **Database Context:**
                - **CTC-related queries** → Use this sample data:
                  {sample_data}
                - **Policy-related queries** → Retrieve relevant policy information dynamically:
                  {relevant_policies}
                
                ## 🔹 **Response Format**
                CLASSIFICATION: <POLICIES / ORACLE / CTC / NONE>
                REFINED_QUESTION: <Improved question or follow-up for clarification>
                ERROR: <Error message if unclear, else empty string>
                CHART_TYPES: <List of chart types, e.g., bar_chart, line_chart, etc. If no chart is needed then empty list>
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """Database Schema:
                {schema_info}
                
                Chat History:
                {chat_history}
                
                User Question:
                {question}
                
                Analyze this question according to the all guidelines."""
            )
        ])

        self.error_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """Given the following user question and error message, generate a conversational response to help the user understand their mistake and provide a helpful follow-up question or suggestion.

                Question: {question}
                Error Message: {error_message}
                
                Provide the response in this JSON format without any additional text:
                {{
                    "title": "<a concise title summarizing the error>",
                    "text_answer": "<a conversational plain text explanation of the error and follow-up question>",
                    "html_answer": "<a conversational HTML-formatted explanation of the error and follow-up question>"
                }}
                """
            )
        ])

    async def analyze_question(
        self,
        question: str,
        memory: ConversationBufferMemory,
        schema_info: Optional[str] = None,
        sample_data: Optional[List[Dict]] = None,
        relevant_policies: Optional[str] = None
    ) -> AnalysisResult:
        chat_history = self.token_service.summarize_chat_history(memory)
        max_context_tokens = self.token_service.calculate_context_limit(question, chat_history)
        
        # optimized_schema_info = self.token_service.optimize_context(schema_info or "", max_context_tokens // 3)
        optimized_sample_data = self._format_sample_data(sample_data, 10)
        # optimized_policies = self.token_service.optimize_context(relevant_policies)

        # return {
        #     'optimized_schema_info': schema_info,
        #     'optimized_sample_data': optimized_sample_data,
        #     'optimized_policies': relevant_policies
        # }

        relevant_policies = f""" Group Risk Management Policy - ....."""
        
        chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        response = await chain.arun({
            "question": question,
            "chat_history": chat_history,
            "schema_info": schema_info,
            "sample_data": optimized_sample_data,
            "relevant_policies": relevant_policies
        })
        
        return self._parse_response(response)
    

    async def generate_conversational_error(self, question: str, error_message: str) -> Dict:
        chain = LLMChain(llm=self.llm, prompt=self.error_prompt)
        response = await chain.arun({
            "question": question,
            "error_message": error_message
        })
        
        try:
            conversational_error = eval(response)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            conversational_error = {
                "title": response.split("\n")[0],
                "text_answer": response,
                "html_answer": response.replace("\n", "<br>")
            }
        
        return conversational_error

    @staticmethod
    def _format_sample_data(data: List[Dict], max_rows: int = 2) -> str:
        if not data:
            return ""
        
        formatted_rows = []
        for i, row in enumerate(data[:max_rows], 1):
            items = [f"{k}: {v:,}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in row.items()]
            formatted_rows.append(f"Row {i}: " + ", ".join(items))
        
        return "\n".join(formatted_rows)

    @staticmethod
    def _parse_response(response: str) -> AnalysisResult:
        result = AnalysisResult(classification="", refined_question="", error="", chart_types=[])
        
        for line in response.strip().split('\n'):
            if line.startswith('CLASSIFICATION:'):
                result.classification = line.replace('CLASSIFICATION:', '').strip().upper()
            elif line.startswith('REFINED_QUESTION:'):
                result.refined_question = line.replace('REFINED_QUESTION:', '').strip()
            elif line.startswith('ERROR:'):
                result.error = line.replace('ERROR:', '').strip()
            elif line.startswith('CHART_TYPES:'):
                chart_types = line.replace('CHART_TYPES:', '').strip()
                if chart_types.lower() not in ['none', 'n/a', '']:
                    result.chart_types = [t.strip() for t in chart_types.split(',') if t.strip()]
        
        return result
