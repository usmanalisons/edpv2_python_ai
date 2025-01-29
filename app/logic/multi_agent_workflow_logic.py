# from typing import Annotated, Sequence, TypedDict, Union, List
# from langgraph.graph import Graph, MessageGraph
# from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage  
# from langchain_core.prompts import PromptTemplate
# from langchain_core.prompt_values import PromptValue
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel
# import operator
# from enum import Enum
# from uuid import uuid4

# from app.services.sql_db_service import SQLDatabaseService
# from app.utils.helper import Helper


# class QueryState(TypedDict):
#     messages: List[BaseMessage]
#     next_step: str
#     current_data: dict
#     query_context: dict


# class IntentType(str, Enum):
#     CLARIFICATION_NEEDED = "clarification_needed"
#     EXECUTE_QUERY = "execute_query"
#     GENERATE_CHARTS = "generate_charts"
#     GENERATE_TEXT = "generate_text"


# class MultiAgentWorkflow:
#     def __init__(self, model_name: str = "gpt-4", api_key: str = None):
#         self.llm = ChatOpenAI(model=model_name, api_key=api_key)
#         self.sql_service = SQLDatabaseService(connection_name="ctc")
#         self.tables = [
#             "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE",
#             "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE_BY_MONTH",
#             "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE_BY_CATEGORY",
#         ]
#         self.db_name = "ctc_db.dbo"
#         self.rights_table = "VIEW_COMPANY_RIGHTS"
#         self.setup_agents()

#     def setup_agents(self):
#         # Set up the intent analysis agent
#         self.intent_analyzer = self.create_intent_analyzer()

#         # Set up the SQL generation agent
#         self.sql_generator = self.create_sql_generator()

#         # Set up the response formatter agent
#         self.response_formatter = self.create_response_formatter()

#         # Create the workflow graph
#         self.workflow = self.create_workflow()

#     def create_intent_analyzer(self):
#         system_message = SystemMessage(
#             content="""You are an expert at analyzing user queries about database information. 
#             Determine if the query is clear enough to execute or needs clarification.
#             Available tables: {tables}
            
#             Return one of these intents:
#             - clarification_needed: If the query is ambiguous or needs more information
#             - execute_query: If the query is clear and can be executed
#             - generate_charts: If the query explicitly requests visual representation
#             - generate_text: If the query just needs data in text format
            
#             Also identify which tables might be relevant."""
#         )

#         intent_prompt = PromptTemplate(input_variables=["query", "tables"], template=system_message.content)

#         return self.llm.bind(
#             prompt=intent_prompt,
#             function_call={"name": "analyze_intent"},
#             functions=[
#                 {
#                     "name": "analyze_intent",
#                     "description": "Analyze the query intent",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "intent": {
#                                 "type": "string",
#                                 "enum": [i.value for i in IntentType],
#                             },
#                             "relevant_tables": {
#                                 "type": "array",
#                                 "items": {"type": "string"},
#                             },
#                             "clarification_needed": {
#                                 "type": "string",
#                                 "description": "If clarification is needed, what should we ask?",
#                             },
#                         },
#                         "required": ["intent", "relevant_tables"],
#                     },
#                 }
#             ],
#         )

#     def create_sql_generator(self):
#         system_message = SystemMessage(
#             content="""Generate a SQL query based on the user request.
#             Use these tables: {tables}
#             Database: {db_name}
#             Rights table: {rights_table}
            
#             Important:
#             - Always include rights check using {rights_table}
#             - Sanitize the query
#             - Use appropriate JOINs
#             - Consider performance"""
#         )

#         sql_prompt = PromptTemplate(input_variables=["query", "tables", "db_name", "rights_table"], template=system_message.content)

#         return self.llm.bind(
#             prompt=sql_prompt,
#             function_call={"name": "generate_sql"},
#             functions=[
#                 {
#                     "name": "generate_sql",
#                     "description": "Generate SQL query",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "sql_query": {"type": "string"},
#                             "explanation": {"type": "string"},
#                         },
#                         "required": ["sql_query"],
#                     },
#                 }
#             ],
#         )

#     def create_response_formatter(self):
#         system_message = SystemMessage(
#             content="""Format the query results based on the user's request.
#             Consider:
#             - If charts were requested, suggest appropriate visualizations
#             - For text responses, structure the data clearly
#             - Highlight key insights
#             - Use appropriate formatting"""
#         )

#         format_prompt = PromptTemplate(input_variables=["query", "data"], template=system_message.content)

#         return self.llm.bind(
#             prompt=format_prompt,
#             function_call={"name": "format_response"},
#             functions=[
#                 {
#                     "name": "format_response",
#                     "description": "Format the response",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "formatted_text": {"type": "string"},
#                             "charts_needed": {
#                                 "type": "array",
#                                 "items": {"type": "string"},
#                             },
#                             "insights": {"type": "array", "items": {"type": "string"}},
#                         },
#                         "required": ["formatted_text"],
#                     },
#                 }
#             ],
#         )

#     async def analyze_intent_step(self, state: QueryState) -> QueryState:
#         """Analyze the user's query intent"""
#         query = state["messages"][-1].content

#         # Format the input as a PromptValue
#         formatted_input = PromptValue({"query": query, "tables": ", ".join(self.tables)})

#         # Call the LLM with the formatted prompt
#         result = await self.intent_analyzer.acall(formatted_input)

#         # Parse the function call result
#         intent_data = result.get("function_call", {}).get("arguments", {})

#         state["query_context"] = intent_data
#         state["next_step"] = intent_data.get("intent", "clarification_needed")
#         return state

#     async def generate_sql_step(self, state: QueryState) -> QueryState:
#         """Generate SQL query based on the intent"""
#         if state["next_step"] != IntentType.EXECUTE_QUERY:
#             return state

#         query = state["messages"][-1].content

#         # Format the input as a PromptValue
#         formatted_input = PromptValue(
#             {
#                 "query": query,
#                 "tables": ", ".join(self.tables),
#                 "db_name": self.db_name,
#                 "rights_table": self.rights_table,
#             }
#         )

#         # Call the LLM with the formatted prompt
#         result = await self.sql_generator.acall(formatted_input)

#         # Parse the function call result
#         sql_data = result.get("function_call", {}).get("arguments", {})

#         # Execute the SQL query
#         sql_query = sql_data.get("sql_query")
#         data = self.sql_service.run_query(sql_query)
#         state["current_data"] = {"raw_results": data, "sql_query": sql_query}
#         state["next_step"] = IntentType.GENERATE_TEXT
#         return state

#     async def format_response_step(self, state: QueryState) -> QueryState:
#         """Format the response based on the data and intent"""
#         query = state["messages"][-1].content

#         # Format the input as a PromptValue
#         formatted_input = PromptValue(
#             {
#                 "query": query, 
#                 "data": str(state["current_data"].get("raw_results", [])),
#             }
#         )

#         # Call the LLM with the formatted prompt
#         result = await self.response_formatter.acall(formatted_input)

#         # Parse the function call result
#         format_data = result.get("function_call", {}).get("arguments", {})
        
#         # Generate any requested charts
#         if format_data.get("charts_needed"):
#             charts = {}
#             for chart_type in format_data["charts_needed"]:
#                 chart_path = Helper.plot_chart(
#                     state["current_data"]["raw_results"],
#                     chart_type=chart_type,
#                     output_dir="app/charts",
#                     chart_name_prefix=chart_type,
#                 )
#                 public_url = (
#                     chart_path.replace("app/charts", "http://localhost:8001")
#                     .replace("\\", "/")
#                 )
#                 charts[chart_type] = public_url
            
#             state["current_data"]["charts"] = charts

#         state["current_data"]["formatted_response"] = format_data.get(
#             "formatted_text", ""
#         )
#         state["next_step"] = "complete"
#         return state

#     def create_workflow(self) -> Graph:
#         """Create the workflow graph"""
#         workflow = Graph()

#         # Add nodes
#         workflow.add_node("analyze_intent", self.analyze_intent_step)
#         workflow.add_node("generate_sql", self.generate_sql_step)  
#         workflow.add_node("format_response", self.format_response_step)

#         # Add edges
#         workflow.add_edge("analyze_intent", "generate_sql")
#         workflow.add_edge("generate_sql", "format_response")

#         # Set entry point
#         workflow.set_entry_point("analyze_intent")

#         return workflow.compile()

#     async def process_query(self, query: str, user_email: str, chat_id: str = None) -> dict:
#         """Process a user query through the workflow"""
#         if not chat_id:
#             chat_id = str(uuid4())

#         # Initialize state with proper message format
#         initial_message = HumanMessage(content=query)
#         state = QueryState(
#             messages=[initial_message],
#             next_step="start",
#             current_data={},
#             query_context={"user_email": user_email},
#         )

#         # Run the workflow
#         final_state = await self.workflow.ainvoke(state)

#         # Prepare response
#         response = {
#             "data": {
#                 "type": "text",
#                 "charts": final_state["current_data"].get("charts", {}),
#                 "answer": final_state["current_data"].get("formatted_response", ""),
#                 "sql_query": final_state["current_data"].get("sql_query", ""),
#                 "sql_results": final_state["current_data"].get("raw_results", []),
#                 "title": "CTC_SEARCH",
#                 "chat_id": chat_id,
#             }
#         }

#         return response