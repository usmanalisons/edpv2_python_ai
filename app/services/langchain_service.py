# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from app.core.config import settings

# from langchain_openai.chat_models import ChatOpenAI
# from langchain_openai.embeddings import OpenAIEmbeddings

# class LangChainService:
#     def __init__(self, model_name: str = None, embedding_model: str = None):
#         self.model_name = model_name if model_name else "gpt-4"
#         self.embedding_model = embedding_model if embedding_model else "text-embedding-ada-002"
#         self.llm = ChatOpenAI(model=self.model_name, openai_api_key=settings.OPENAI_API_KEY, temperature=0.7)
#         self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model=self.embedding_model)
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     def get_embeddings(self, texts):
#         if isinstance(texts, str):
#             texts = [texts]
#         return self.embeddings.embed_documents(texts)
    
#     def generate_contextual_answer(self, query: str, context: str) -> str:
#         prompt_template = PromptTemplate(
#         input_variables=["query", "context"],
#         template="""
#             You are an AI assistant.
#             - Do not provide any additional context, explanations, or information beyond what is strictly necessary for the query.
#             - Always format your answer in HTML.

#             Query: {query}
#             Context: {context}

#             Answer in HTML:
#             """
#         )
#         print('Memory: ', self.memory.chat_memory.messages)
#         chain = LLMChain(llm=self.llm, prompt=prompt_template,)
#         response = chain({"query": query, "context": context})
#         return response["text"].strip()


#     def generate_sql_query(self, user_query: str, schema_description: str, table_names: list, database: str, rights_schema: str, user: dict, rights_table: str) -> str:
#         user_info_str = "\n".join([f"{k}: {v}" for k, v in user.items()])
#         prompt_template = PromptTemplate(
#             input_variables=["user_query", "schema_description", "table_names", "database", "rights_schema", "user_info", "rights_table"],
#             template="""
#             You are an AI assistant that converts natural language queries into SQL Server SELECT statements.
#             The database is '{database}'.
#             The following tables/views are available: {table_names}, rights table: {rights_table}
#             Schemas:
#             {schema_description}

#             Rights Schema:
#             {rights_schema}

#             User Information:
#             {user_info}

#             Requirements:
#             - If the query is conversational, respond conversationally (no SQL).
#             - If informational, return a valid SQL SELECT statement only.
#             - Use WHERE with LIKE '%...%' instead of '=' for string comparisons.
#             - No extra explanations or code fences.
#             - Enforce user rights as described.
            
#             Column Selection Rules:
#             1. Essential Columns (Always Include):
#             - Primary identifiers (project_id, task_id, etc.)
#             - Key dates (as date_column alias)
#             - Main metrics requested in the query
            
#             2. Supporting Columns (Include when relevant):
#             - Category/grouping fields needed for context
#             - Comparison fields (previous period, variances)
#             - Status indicators
            
#             3. Column Formatting:
#             - Always use meaningful column aliases (e.g., 'Total_Cost' instead of 'sum(cost)')
#             - Format dates as 'YYYY-MM-DD'
#             - Round numerical values to 2 decimal places using ROUND()
#             - Use COALESCE() for potentially null values
            
#             4. Column Order:
#             - Identifiers first
#             - Time periods second
#             - Metrics and calculations last
#             - Group related columns together
            
#             5. Time-Based Queries:
#             - Include both period start and end dates when showing ranges
#             - Use DATEPART(QUARTER, date_column) for quarterly grouping
#             - Use DATEPART(YEAR, date_column) for yearly grouping
#             - For monthly to quarterly conversion, use appropriate aggregation functions
#             - Include both period start and end dates when showing ranges if needed
#             6 - For comparisons:
#             - Use LAG() or LEAD() for previous/next period comparisons
#             - Use appropriate window functions for running totals or moving averages
            
#             Example Column Pattern:
#             SELECT
#                 p.project_id AS Project_ID,
#                 p.project_name AS Project_Name,
#                 FORMAT(p.date_column, 'yyyy-MM-dd') AS Report_Date,
#                 ROUND(p.primary_metric, 2) AS Primary_Metric,
#                 COALESCE(p.secondary_metric, 0) AS Secondary_Metric,
#                 -- Additional metrics as needed

#             Query validation:
#             - If the user requests a non-existent column or table, respond with "Invalid query: [reason]."
#             - If user lacks access based on the rights table, respond with "Access denied."

#             User query: "{user_query}"
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)

#         response = chain.run(
#             user_query=user_query,
#             schema_description=schema_description,
#             table_names=", ".join(table_names),
#             database=database,
#             rights_schema=rights_schema,
#             user_info=user_info_str,
#             rights_table=rights_table
#         )

#         return response.strip()
