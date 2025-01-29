# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai.chat_models import ChatOpenAI
# from app.utils.helper import Helper
# from langchain.schema import HumanMessage, AIMessage
# import re
# import json
# import logging
# from decimal import Decimal
# from typing import Tuple, List

# class DatabaseSearchPromptService:
#     def __init__(self, model_name: str, api_key: str):
#         self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.7)

#     @staticmethod
#     def summarize_results(rows: list) -> str:
#         if not rows:
#             return "No data returned available."
        
#         column_names = list(rows[0].keys())
#         summary = f"Columns: {', '.join(column_names)}\n"

#         for idx, row in enumerate(rows, start=1):
#             row_data = ", ".join([f"{key}: {value}" for key, value in row.items()])
#             summary += f"Row {idx}: {row_data}\n"

#         return summary


#     def analyze_question_intent(self, question: str, tables_info: str, memory=None):
#         chat_history = self.summarize_messages(memory.chat_memory.messages, max_messages=10) if memory else "No history."

#         previous_human_messages = []
#         if memory and memory.chat_memory.messages:
#             for msg in memory.chat_memory.messages:
#                 if isinstance(msg, HumanMessage):
#                     previous_human_messages.append(msg.content)

#         numbered_human_messages = "\n".join(
#             [f"{idx + 1}. {message}" for idx, message in enumerate(previous_human_messages)]
#         )

#         merged_input = f"""
#         The database has the following tables:
#         {tables_info}
#         Chat history (for context):
#         {chat_history}
#         The user's current question is: {question}
#         Previous human messages (for reference):
#         {numbered_human_messages}
#         """
#         input_tokens = Helper.count_tokens(merged_input)
#         max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
#         self.llm.max_tokens = min(max_completion_tokens, 16384)

#         prompt_template = PromptTemplate(
#             input_variables=["user_question", "tables_info", "chat_history", "previous_human_messages"],
#             template="""
#                 You are an AI assistant that processes user queries for generating SQL queries or understanding user intent.
                
#                 The database has the following tables:
#                 {tables_info}
#                 Chat history (for context):
#                 {chat_history}
#                 The user's current question is: {user_question}
#                 Previous human messages (for reference):
#                 {previous_human_messages}

#                 Your Task:
#                 1. Determine the user's intent based on their current question and the context from chat history, previous human messages, and relevant tables.
#                 2. If the question refers to any previous search, dynamically refine it by linking it to the most relevant question from the previous human messages.
#                 3. If the question is valid, determine if the user wants textual results or a chart. Map chart requests to: "bar_chart", "line_chart", "pie_chart", "column_chart".
#                 4. If the question involves both history and database data, classify it as "get_data_from_tables" intent.
#                 5. If the question is unclear or nonsensical, return "unclear" along with a conversational explanation of what is missing.

#                 Return your response in this format:
#                 - intents: A comma-separated list of identified intents without any extra space or quotes.
#                 - message: A conversational explanation if unclear without any extra space or quotes; otherwise, an empty string.
#                 - refined_question: A single refined question combining the user's current question and relevant historical context without any extra space or quotes.
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         raw_response = chain.run({
#             "user_question": question,
#             "tables_info": tables_info,
#             "chat_history": chat_history,
#             "previous_human_messages": numbered_human_messages
#         }).strip()

#         intents_match = re.search(r"- intents: (.+)", raw_response)
#         message_match = re.search(r"- message: (.+)", raw_response)
#         refined_question_match = re.search(r"- refined_question: (.+)", raw_response)

#         intents = intents_match.group(1).strip().split(",") if intents_match else []
#         error_message = message_match.group(1).strip() if message_match else ""
#         refined_question = refined_question_match.group(1).strip() if refined_question_match else question

#         if error_message:
#             return ([], error_message, refined_question)
#         else:
#             return (list(set(intents)), "", refined_question)


#     def check_question_vagueness(self, question: str, tables_info: str, memory=None):
#         chat_history = self.summarize_messages(memory.chat_memory.messages, max_messages=10) if memory else "No history."

#         merged_input = f"""
#         The database has the following tables:
#         {tables_info}
#         Chat history (for context):
#         {chat_history}
#         The user's question is: {question}
#         """
#         input_tokens = Helper.count_tokens(merged_input)
#         max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
#         self.llm.max_tokens = min(max_completion_tokens, 16384)

#         prompt_template = PromptTemplate(
#             input_variables=["question", "tables_info", "chat_history"],
#             template="""
#                 You are an AI assistant tasked with evaluating user queries for clarity and specificity.

#                 The database has the following tables:
#                 {tables_info}
#                 Chat history (for context):
#                 {chat_history}
#                 The user's question is: {question}

#                 Your Task:
#                 1. Determine whether the question is specific and actionable given the provided tables and chat context.
#                 2. If the question not clear, incomplete, or lacks actionable details, provide a conversational explanation of why it is unclear and suggest how the user can make it clearer.
#                 3. If the question is clear and actionable, respond with an empty explanation.
#                 4. User can ask about some graphs or visualizations which are valid questions so don't return error message for that
#                 5. Don't mention table, error occurred, error found as text or table names in the error_messages, just a clear error_message for the user because user does not know about the tables or columns

#                 Strictly return your response in this format:
#                 error_message: A conversational response back to the user if the user question is vague, or an empty string if it is clear. Do not add anything else to the response no extra space, quotes or slashes.
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         raw_response = chain.run({
#             "question": question,
#             "tables_info": tables_info,
#             "chat_history": chat_history
#         }).strip()


#         print("raw_response: ", raw_response)

#         if raw_response.startswith("error_message:"):
#             error_message = raw_response[len("error_message:"):].strip()
#         else:
#             error_message = "Unexpected response format. Please refine your question."

#         return error_message



#     def extract_data_from_history(self, question: str, memory):
        
#         chat_history = self.summarize_messages(memory.chat_memory.messages, max_messages=10) if memory else "No history."
#         chat_history = self.clean_chat_history(chat_history)

#         prompt_template = PromptTemplate(
#             input_variables=["user_question", "chat_history"],
#             template="""
#                 You are an AI assistant tasked with extracting structured JSON data from chat history.
#                 Chat history (cleaned for your reference):
#                 {chat_history}
#                 User question:
#                 {user_question}
#                 Your task:
#                 - Search the chat history for relevant information based on the user's question.
#                 - Return the information in the following JSON format don't add any extra space, quotes just a clean JSON:
#                 [
#                     {{
#                         "key": "<key>",
#                         "value": "<value>"
#                     }},
#                     ...
#                 ]
#                 If no relevant information is found, return:
#                     []
#                 Ensure the JSON is properly formatted and valid without any extra space or quotes.
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         raw_response = chain.run({
#             "user_question": question,
#             "chat_history": chat_history
#         }).strip()

#         try:
#             structured_data = json.loads(raw_response)
#         except json.JSONDecodeError:
#             logging.error(f"Malformed JSON response: {raw_response}")
#             structured_data = []

#         return structured_data

#     def clean_chat_history(self, chat_history: str) -> str:
#         chat_history = re.sub(r"<[^>]*>", "", chat_history)
#         chat_history = re.sub(r"\s+", " ", chat_history).strip()
#         return chat_history[:5000]
    

#     def _get_chart_template(self, chart_type: str) -> str:
#         templates = {
#             "bar_chart": """
#                 For bar charts:
#                 - X-axis should contain categorical data
#                 - Y-axis must contain numerical values
#                 - Sort bars by y-axis values if not time-based
#                 - Return data ordered by y-values in descending order for better visualization
#                 - Limit to top 15 categories if there are too many
#                 - without any extra space or quotes
#             """,
#             "line_chart": """
#                 For line charts:
#                 - X-axis should preferably be temporal or sequential
#                 - Y-axis must contain numerical values
#                 - Sort data points by x-axis values
#                 - Ensure x-axis values are properly ordered
#                 - If dates are present, format them consistently
#                 - without any extra space or quotes
#             """,
#             "pie_chart": """
#                 For pie charts:
#                 - X-axis should contain categories (labels)
#                 - Y-axis must contain positive numerical values
#                 - Consider grouping small values into "Other" category
#                 - Limit to top 8 categories, grouping remainder into "Other"
#                 - Sort values in descending order
#                 - without any extra space or quotes
#             """
#         }
#         return templates.get(chart_type, "")

#     def prepare_data_for_chart(self, json_data: list, chart_type: str, user_question: str):
#         def convert_decimal(obj):
#             if isinstance(obj, Decimal):
#                 return float(obj)
#             raise TypeError(f"Type {type(obj)} not serializable")

#         prompt_template = PromptTemplate(
#             input_variables=["json_data", "chart_type", "user_question"],
#             template="""
#                 You are an AI assistant that transforms raw JSON data into structured rows for generating charts using matplotlib.
#                 The JSON data provided is:
#                 {json_data}
#                 The user question is:
#                 {user_question}
#                 The user wants to generate a {chart_type}. Your task is:
#                 1) Identify the best columns for the x-axis and y-axis based on the chart type and user question.
#                 2) Ensure that the x-axis contains labels or sequential values, and the y-axis contains numerical values.
#                 3) Ignore rows with missing, null, or invalid data for the selected x and y columns.
#                 4) Return ONLY a list of dictionaries with this EXACT structure (do not include any other text and without any extra space or quotes):
#                 [
#                     {{"x_col": <x_value>, "y_col": <y_value>}},
#                     ...
#                 ]
#             """
#         )

#         try:
#             json_data_serialized = json.dumps(json_data, indent=2, default=convert_decimal)

#             chain = LLMChain(llm=self.llm, prompt=prompt_template)
#             response = chain.run({
#                 "json_data": json_data_serialized,
#                 "chart_type": chart_type,
#                 "user_question": user_question
#             }).strip()

#             response = response.replace('```json', '').replace('```', '').strip()

#             start_idx = response.find('[')
#             end_idx = response.rfind(']') + 1
#             if start_idx != -1 and end_idx > 0:
#                 json_str = response[start_idx:end_idx]

#                 parsed_data = json.loads(json_str)
#                 if not isinstance(parsed_data, list):
#                     return []

#                 formatted_data = []
#                 for item in parsed_data:
#                     if isinstance(item, dict):
#                         formatted_item = {
#                             "x_col": item.get('"x_col"') or item.get('x_col'),
#                             "y_col": item.get('"y_col"') or item.get('y_col')
#                         }
#                         if formatted_item["x_col"] is not None and formatted_item["y_col"] is not None:
#                             formatted_data.append(formatted_item)

#                 return formatted_data

#             return []

#         except Exception as e:
#             print(f"Error preparing data: {str(e)}")
#             return []

#     def generate_final_answer(
#         self, 
#         user_question: str, 
#         sql_results: list, 
#         intents: list, 
#         generated_charts: dict = None, 
#         generate_title: bool = True, 
#         error_message: str = None
#     ):
#         if error_message:
#             results_summary = ""
#             chart_descriptions = ""
#         else:
#             results_summary = self.summarize_results(rows=sql_results)
#             chart_descriptions = "\n".join(
#                 f"{chart_type.replace('_', ' ').title()}: {url}" for chart_type, url in (generated_charts or {}).items()
#             )
        
#         merged_input = f"""
#         User's question: {user_question}
#         {"Error Message: " + error_message if error_message else f"Summary: {results_summary}"}
#         User Intents: {", ".join(intents)}
#         Generated Charts (URLs): 
#         {chart_descriptions}
#         Generate Title: {"true" if generate_title else "false"}
#         """
        
#         input_tokens = Helper.count_tokens(merged_input, model="gpt-4o")
#         max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
#         self.llm.max_tokens = min(max_completion_tokens, 16384)
        
#         prompt_template = PromptTemplate(
#             input_variables=["input"],
#             template="""
#             You are an AI assistant tasked with generating a well-structured HTML response, and optionally, a title.
#             Input Information:
#             {input}
#             Your task:
#             - Use the provided information to create a complete and user-friendly HTML response. 
#             - Format any amounts provided in the summary or results as follows:
#                 - Display them in AED currency format.
#                 - Use comma-separated formatting for large numbers (e.g., 1,234,567.89 AED).
#             - Don't add <h1>, <h2>, <html>, <body>, header, or footer tags, and do not include CSS in the response.
#             - If you have to show data in tabuler format then properly add the borders for it
#             - Include the results summary as a section in the HTML if no error message is provided.
#             - Embed the provided chart URLs as <img> tags under appropriate headings, if available.
#             - If "Generate Title" is true, create a concise and unique title based only on the user's question or error message.
#             Otherwise, set the title to an empty string.
#             - If an error message is provided, ignore the results summary and charts, and base the response solely on the error message.
#             Return your response in the following format:
#             Title: <generated title or "">
#             HTML: <HTML response, don't add any quotes, extra spaces or extra words just a clean HTML>
#             REMEMBER only add the summary data in table when necessary based on the user question
#             """
#         )
        
#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         response = chain.run({"input": merged_input}).strip()
#         try:
#             title_start = response.index("Title:") + len("Title:")
#             html_start = response.index("HTML:")
#             title = response[title_start:html_start].strip()
#             html = response[html_start + len("HTML:"):].strip()
#         except ValueError:
#             title = ""
#             html = response

#         return html, title
    
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


#     @staticmethod
#     def get_relevant_context(context, max_contexts=3):
#         context_list = context.split(". ") 
#         return ". ".join(context_list[:max_contexts]) 
    

#     def generate_sql_query(self, user_question: str, schema_description: str, table_names: list, database: str, rights_schema: str, user_email: str, rights_table: str, distinct_data: list):
#         user_info_str = f"user email = {user_email}"
        
#         ctc_data_str = "\n".join([
#             f"Company: {record['COMPANY_NAME']} (Code: {record['COMPANY_CODE']}), "
#             f"Project: {record['PROJECT_NAME']} (Code: {record['PROJECT_CODE']}), "
#             f"CTC Status: {record['CTC_STATUS']}, "
#             f"Total Budget: {record['TOTAL_BUDGET']}, Total Cost at Completion: {record['TOTAL_COST_AT_COMPLETION']}, "
#             f"Variation at Completion: {record['VARIATION_AT_COMPLETION']}"
#             for record in distinct_data
#         ])
        
#         prompt_template = PromptTemplate(
#             input_variables=["user_question", "schema_description", "table_names", "database", "rights_schema", "user_info", "rights_table", "latest_ctc_data", "user_email"],
#             template="""
#             You are an advanced SQL assistant that generates valid and optimized SQL Server `SELECT` queries based on user requirements.

#             ### Context ###
#             - Database: '{database}'
#             - Available Tables/Views: {table_names}
#             - Rights Table: {rights_table}

#             ### Schema Information in XML format ###
#             {schema_description}

#             ### User Rights ###
#             - Rights Schema in string format: {rights_schema}
#             - User Information: {user_info}

#             ### Latest CTC Records ###
#             - Example records (distinct by company): {latest_ctc_data}

#             ### Task ###
#             1. **Objective**: Generate a valid, optimized SQL Server `SELECT` query to answer the user's question: "{user_question}".
#             2. **Strict Output Requirements**:
#             - The output must be a valid SQL Server `SELECT` query only, starting with `SELECT` and containing no extra text, comments, or explanations.
#             - The query must be clean and formatted properly with no extraneous text or commentary.
#             3. **Query Construction Rules**:
#             - Use only columns explicitly defined in the schema. Do not invent or assume columns.
#             - Dynamically identify and include required `JOIN`s based on column dependencies or relationships in the schema.
#             - Fully qualify all column names with their respective table or alias to avoid ambiguity.
#             - Enforce user rights by joining the rights table on `COMPANY_ID` and filtering rows with `COMPANY_USER_MAIL LIKE '%{user_email}%'`.
#             - Use `LIKE '%...%'` instead of `=` for string comparisons.
#             - Ensure null safety by using `NULLIF` for division operations to prevent divide-by-zero errors.
#             - Avoid invalid constructs, such as nested aggregate functions (e.g., `SUM(SUM(...))`). Break them into independent calculations instead.
#             - If using `DISTINCT` in combination with `ORDER BY`, ensure all `ORDER BY` columns are included in the `SELECT` list.
#             4. **Error Prevention**:
#             - Validate all column references against the schema. Do not include columns that are not explicitly part of the schema.
#             - If a column does not exist in the schema, exclude it and provide a simplified query that still answers the user's question.
#             - Avoid ambiguous column references and ensure all aliases are properly declared and used.

#             ### Dynamic Adjustments ###
#             - If the question is vague (e.g., missing timeframes or metrics), assume reasonable defaults:
#             - Use all available data if no time period is specified.
#             - Compare using absolute differences unless otherwise stated.
#             """
#         )

#         chain = LLMChain(llm=self.llm, prompt=prompt_template)
#         response = chain.run(
#             user_question=user_question,
#             schema_description=schema_description,
#             table_names=", ".join(table_names),
#             database=database,
#             rights_schema=rights_schema,
#             user_info=user_info_str,
#             rights_table=rights_table,
#             latest_ctc_data=ctc_data_str,
#             user_email=user_email,
#         )
        
#         response = response.strip()
#         response = response.lstrip("`").rstrip("`").strip()
        
#         if response.lower().startswith("sql"):
#             response = response[response.lower().index("select"):].strip()
#         response = response.replace("`", "").strip()

#         return response







from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import json
import logging
from app.services.token_service import TokenService

class DatabaseSearchPromptService:
    """
    A service for handling database queries, SQL generation, and response formatting.
    This service combines query analysis, SQL generation, and response formatting
    while maintaining efficient token usage and clear conversation structure.
    """
    
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0.7
        )
        self.token_service = TokenService(model_name=model_name)


        self.merged_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an advanced AI assistant that analyzes user queries in multiple ways:
                1. Classification: Determine which system should handle the query
                2. Query Refinement: Either improve the query or request clarification
                3. Error Detection: Identify if the query is unclear or incomplete
                4. Visualization Needs: Determine if the query requires any charts or visualizations

                **Classification Guidelines:**
                - POLICIES: Company policies, procedures, guidelines, regulations, compliance, loan, security
                - ORACLE: Oracle systems, training materials, Oracle-related procedures
                - CTC: Project costs, budgets, cost estimates, financial data
                - EMPTY: When the query is unclear or needs clarification

                **Refinement Guidelines:**
                - For clear queries: Enhance them for better search results
                - For unclear queries: Create a user-friendly follow-up question that:
                  * Clearly explains what additional information is needed
                  * Uses natural, conversational language
                  * Provides examples when helpful
                  * Avoids technical jargon

                **Visualization Guidelines:**
                - Identify if the query requires any of the following chart types:
                  * bar_chart
                  * line_chart
                  * pie_chart
                  * column_chart

                **Error Handling:**
                - If the query is unclear or incomplete, provide an error message explaining the issue.

                **Database Context:**
                - Here is some sample data from the database to help you understand the terms and structure:
                {sample_data}

                **Response Format:**
                CLASSIFICATION: <POLICIES/ORACLE/CTC/EMPTY>
                REFINED_QUERY: <refined query or follow-up question>
                ERROR: <error message if unclear, empty otherwise>
                CHART_TYPES: <comma-separated list of requested chart types if any>"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Database Schema:
                {tables_info}

                Chat History:
                {chat_history}

                User Question:
                {question}

                Analyze this query according to the guidelines."""
            )
        ])

        
        # Initialize the SQL generation prompt template
        self.sql_generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an SQL expert generating optimized SQL Server queries.
                Follow these strict rules:
                1. Use only columns from the provided schema
                2. Fully qualify all column names (table.column)
                3. Include proper JOINs and user rights filtering
                4. Ensure NULL safety in calculations
                5. Avoid nested aggregates
                6. Use LIKE with wildcards for string matching
                
                Return only the SQL query, no explanations."""
            ),
            HumanMessagePromptTemplate.from_template(
                """Database: {database}
                Tables: {table_names}
                Schema: {schema_description}
                Rights: {rights_schema}
                User: {user_email}
                Rights Table: {rights_table}
                Sample Data: {sample_data}
                
                Question: {question}"""
            )
        ])
        
        # Initialize the response generation prompt template
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """Your task is to analyze the provided data and generate a response that directly answers the user's question while providing context, insights, and visualizations.

                **Response Requirements:**
                1. **Text Response:**
                - Start with a concise summary of the key findings.
                - Explain the data in detail, addressing the user's question directly.
                - Highlight trends, patterns, or anomalies in the data.
                - Use bullet points or numbered lists for clarity when appropriate.
                - Format numbers with commas and AED currency (e.g., 1,234,567.89 AED).
                - If applicable, calculate percentages, growth rates, or other metrics to provide deeper insights.

                2. **HTML Response:**
                - Use clean, semantic HTML for structure.
                - Include tables with borders for data presentation.
                - Embed charts as `<img>` tags using base64-encoded images.
                - Use headings (h3, h4, h5) to organize sections.
                - Add descriptive captions for charts and tables.
                - Ensure the HTML is self-contained (no external CSS or scripts).

                3. **Title (if requested):**
                - Generate a clear and descriptive title that summarizes the response.

                4. **Error Handling:**
                - If an error is provided, explain it clearly and suggest possible solutions or next steps.

                **Example Response:**
                TEXT: The analysis shows that the ASMEF company has 100 projects distributed across three risk categories: High Risk (20 projects), Medium Risk (50 projects), and Low Risk (30 projects). The majority of projects fall under the Medium Risk category, indicating a balanced risk profile. However, the High Risk projects require immediate attention to mitigate potential issues.

                HTML: <h3>Project Risk Analysis for ASMEF</h3>
                    <p>The analysis shows the following distribution of projects by risk category:</p>
                    <table border="1">
                        <thead>
                            <tr><th>Risk Category</th><th>Project Count</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>High Risk</td><td>20</td></tr>
                            <tr><td>Medium Risk</td><td>50</td></tr>
                            <tr><td>Low Risk</td><td>30</td></tr>
                        </tbody>
                    </table>
                    <div class="chart-container">
                        <h4>Project Distribution by Risk Category</h4>
                        <img src="data:image/png;base64,..." alt="Project Distribution Chart" />
                        <p>This pie chart illustrates the proportion of projects in each risk category.</p>
                    </div>

                TITLE: Project Risk Analysis for ASMEF

                **Return EXACTLY in this format:**
                TEXT: <natural text response>
                HTML: <formatted HTML>
                TITLE: <title if needed>"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Question: {question}
                Data Summary: {data_summary}
                Charts: {charts_info}
                Error: {error_message}
                Generate Title: {generate_title}"""
            )
        ])

    async def analyze_question(
        self,
        question: str,
        tables_info: str,
        memory: ConversationBufferMemory,
        sample_data: List[Dict] = None
    ) -> Dict[str, str]:
        """
        Analyzes a user query to determine classification, refined query, error, and chart types.
        
        Args:
            question (str): The user's query.
            tables_info (str): Information about the database schema.
            memory (ConversationBufferMemory): The chat history memory.
            sample_data (List[Dict]): Sample data from the database to provide context.

        Returns:
            Dict[str, str]: A dictionary containing classification, refined query, error, and chart types.
        """
        # Get optimized chat history using token service
        chat_history = self.token_service.summarize_chat_history(
            memory=memory,
            max_length=1000,
            max_messages=10
        )
        
        # Format sample data for inclusion in the prompt
        formatted_sample_data = self._format_sample_data(sample_data) if sample_data else "No sample data provided."
        
        # Calculate available tokens for the response
        input_text = f"{question}\n{tables_info}\n{chat_history}\n{formatted_sample_data}"
        max_tokens = self.token_service.calculate_max_tokens(input_text)
        
        # Create and run the chain
        chain = LLMChain(
            llm=self.llm.with_config({"max_tokens": max_tokens}),
            prompt=self.merged_analysis_prompt
        )
        
        response = await chain.arun({
            "question": question,
            "tables_info": tables_info,
            "chat_history": chat_history,
            "sample_data": formatted_sample_data
        })
        
        # Parse the response into components
        result = {
            "classification": "",
            "refined_query": "",
            "error": "",
            "chart_types": []
        }
        
        for line in response.strip().split('\n'):
            if line.startswith("CLASSIFICATION:"):
                result["classification"] = line.replace("CLASSIFICATION:", "").strip()
            elif line.startswith("REFINED_QUERY:"):
                result["refined_query"] = line.replace("REFINED_QUERY:", "").strip()
            elif line.startswith("ERROR:"):
                result["error"] = line.replace("ERROR:", "").strip()
            elif line.startswith("CHART_TYPES:"):
                result["chart_types"] = [
                    chart.strip() 
                    for chart in line.replace("CHART_TYPES:", "").strip().split(',')
                    if chart.strip()
                ]
        
        return result

    @staticmethod
    def _format_sample_data(sample_data: List[Dict]) -> str:
        """
        Formats sample database data into a readable string for the prompt.
        
        Args:
            sample_data (List[Dict]): Sample data from the database.

        Returns:
            str: A formatted string representation of the sample data.
        """
        if not sample_data:
            return "No sample data provided."
        
        formatted_data = []
        for row in sample_data:
            formatted_row = ", ".join(f"{key}: {value}" for key, value in row.items())
            formatted_data.append(formatted_row)
        
        return "Sample Data:\n" + "\n".join(formatted_data)


    async def generate_sql(
        self,
        question: str,
        schema_info: Dict,
        user_email: str,
        sample_data: List[Dict]
    ) -> str:
        """
        Generates an optimized SQL query based on the user's question and schema information.
        """
        # Format sample data
        data_summary = "\n".join(
            f"Company: {d['COMPANY_NAME']}, Project: {d['PROJECT_NAME']}, "
            f"Budget: {d['TOTAL_BUDGET']}, Status: {d['CTC_STATUS']}"
            for d in sample_data
        )
        
        # Calculate available tokens
        merged_input = f"{question}\n{str(schema_info)}\n{data_summary}"
        max_tokens = self.token_service.calculate_max_tokens(merged_input)
        
        # Create and run the chain
        chain = LLMChain(
            llm=self.llm.with_config({"max_tokens": max_tokens}),
            prompt=self.sql_generation_prompt
        )
        
        response = await chain.arun({
            "question": question,
            "database": schema_info["database"],
            "table_names": schema_info["tables"],
            "schema_description": schema_info["schema"],
            "rights_schema": schema_info["rights_schema"],
            "rights_table": schema_info["rights_table"],
            "user_email": user_email,
            "sample_data": data_summary
        })
        
        # Clean and format the SQL query
        return self._clean_sql_response(response)

    async def format_response(
        self,
        question: str,
        sql_results: List[Dict],
        charts: Optional[Dict] = None,
        error_message: str = "",
        generate_title: bool = False
    ) -> Dict[str, str]:
        """
        Generates formatted text and HTML responses based on query results and charts.
        """
        # Prepare data summary
        data_summary = self._format_data_summary(sql_results) if sql_results else ""
        
        # Format chart information
        charts_info = "\n".join(
            f"{chart_type}: {url}" 
            for chart_type, url in (charts or {}).items()
        )
        
        # Calculate available tokens
        merged_input = f"{question}\n{data_summary}\n{charts_info}\n{error_message}"
        max_tokens = self.token_service.calculate_max_tokens(merged_input)
        
        # Create and run the chain
        chain = LLMChain(
            llm=self.llm.with_config({"max_tokens": max_tokens}),
            prompt=self.response_prompt
        )
        
        response = await chain.arun({
            "question": question,
            "data_summary": data_summary,
            "charts_info": charts_info,
            "error_message": error_message,
            "generate_title": str(generate_title)
        })
        
        # Parse response into components
        result = {
            "text": "",
            "html": "",
            "title": ""
        }
        
        current_section = None
        current_content = []
        
        for line in response.strip().split('\n'):
            if line.startswith('TEXT:'):
                if current_section:
                    result[current_section.lower()] = '\n'.join(current_content).strip()
                current_section = "text"
                current_content = [line.replace('TEXT:', '').strip()]
            elif line.startswith('HTML:'):
                if current_section:
                    result[current_section.lower()] = '\n'.join(current_content).strip()
                current_section = "html"
                current_content = [line.replace('HTML:', '').strip()]
            elif line.startswith('TITLE:'):
                result["title"] = line.replace('TITLE:', '').strip()
            else:
                if current_section:
                    current_content.append(line)
        
        if current_section:
            result[current_section.lower()] = '\n'.join(current_content).strip()
        
        return result

    async def prepare_chart_data(self, data: List[Dict], question: str, available_charts: List[str]) -> Dict:
        try:
            # First, ensure data is properly serialized
            formatted_data = []
            for row in data:
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        formatted_row[key] = float(value)
                    elif value is None:
                        formatted_row[key] = ""
                    else:
                        formatted_row[key] = value
                formatted_data.append(formatted_row)

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a chart data formatter. Analyze data and format it for visualization."),
                HumanMessage(content=f"""Given this data:
{json.dumps(formatted_data, indent=2)}

For this question:
{question}

Return data formatted for these chart types:
{', '.join(available_charts)}

ONLY return a JSON object with this exact structure (no other text):
{{
    "chart_type_name": [
        {{"x_col": "value", "y_col": number}},
        {{"x_col": "value", "y_col": number}}
    ]
}}""")
            ])

            response = await LLMChain(llm=self.llm, prompt=prompt).arun({})

            # Clean and parse response
            cleaned_response = response.strip()
            if not cleaned_response.startswith('{'):
                cleaned_response = cleaned_response[cleaned_response.find('{'):]
            if not cleaned_response.endswith('}'):
                cleaned_response = cleaned_response[:cleaned_response.rfind('}')+1]

            chart_data = json.loads(cleaned_response)
            
            return {
                chart_type: self._format_chart_data(chart_data.get(chart_type, []))
                for chart_type in available_charts
            }

        except Exception as e:
            logging.error(f"Error preparing chart data: {str(e)}, Data: {data[:2]}, Question: {question}")
            return {chart_type: [] for chart_type in available_charts}

    def _format_chart_data(self, data: List[Dict]) -> List[Dict]:
        formatted_data = []
        for item in data:
            if isinstance(item, dict):
                x_val = item.get('x_col')
                y_val = item.get('y_col')
                if x_val is not None and y_val is not None:
                    try:
                        formatted_data.append({
                            "x_col": str(x_val),
                            "y_col": float(y_val) if isinstance(y_val, (int, float, str, Decimal)) else 0.0
                        })
                    except (ValueError, TypeError):
                        continue
        return formatted_data
        
    @staticmethod
    def _clean_sql_response(sql: str) -> str:
        """Cleans and formats an SQL query response."""
        sql = sql.strip()
        sql = sql.lstrip('`').rstrip('`')
        if sql.lower().startswith('sql'):
            sql = sql[sql.lower().index('select'):].strip()
        return sql.replace('`', '').strip()

    @staticmethod
    def _format_data_summary(data: List[Dict]) -> str:
        """Creates a structured summary of data results."""
        if not data:
            return "No data available."
            
        columns = list(data[0].keys())
        summary = ["Columns: " + ", ".join(columns)]
        
        for idx, row in enumerate(data[:5], 1):  # Show first 5 rows
            row_data = ", ".join(f"{k}: {v}" for k, v in row.items())
            summary.append(f"Row {idx}: {row_data}")
            
        if len(data) > 5:
            summary.append(f"... and {len(data) - 5} more rows")
            
        return "\n".join(summary)