from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from app.utils.helper import Helper
from langchain.schema import HumanMessage, AIMessage
import re
import json
import logging
from decimal import Decimal

class QueryService:
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.7)

    @staticmethod
    def summarize_results(rows: list) -> str:
        if not rows:
            return "No data returned available."
        
        column_names = list(rows[0].keys())
        summary = f"Columns: {', '.join(column_names)}\n"

        for idx, row in enumerate(rows, start=1):
            row_data = ", ".join([f"{key}: {value}" for key, value in row.items()])
            summary += f"Row {idx}: {row_data}\n"

        return summary


    def analyze_query_intent(self, query: str, tables_info: str, memory=None):
        chat_history = self.summarize_messages(memory.chat_memory.messages, max_messages=10) if memory else "No history."
        merged_input = f"""
        The database has the following tables:
        {tables_info}
        Chat history (for context):
        {chat_history}
        The user query is: {query}
        """
        input_tokens = Helper.count_tokens(merged_input)
        max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
        self.llm.max_tokens = min(max_completion_tokens, 16384)

        prompt_template = PromptTemplate(
            input_variables=["user_query", "tables_info", "chat_history"],
            template="""
                You are an AI assistant that tries to figure out:
                1) Whether the user's query is clear enough to proceed with generating a SQL query. Remember, the user can ask for visual graphs or charts as well, which could be valid user queries.
                2) If the user wants a textual result or a chart. Potential chart synonyms: "trend", "visualize", "distribution", "graph", etc. 
                Map them to: - "bar_chart", "line_chart", "pie_chart" , "column_chart", "get_data_from_tables" (if no clear chart is requested).
                3) If the query refers to something in the chat history, summary above data, above response etc, classify it as "get_data_from_history".
                4) If the query involves both history and database data, classify it as "get_data_from_tables" intent.
                The database has the following tables:
                {tables_info}
                Chat history (for context):
                {chat_history}
                The user query is: {user_query}
                If the query is unclear or incomplete, return:
                "unclear"
                Otherwise, return one or more of these:
                "get_data_from_tables", "bar_chart", "line_chart", "pie_chart", "column_chart", "get_data_from_history"
                Remember the the response must contains get_data_from_tables or get_data_from_history along with any requested chart
                Return exactly one line and nothing else.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        raw_response = chain.run({
            "user_query": query,
            "tables_info": tables_info,
            "chat_history": chat_history
        }).strip().lower()

        if "unclear" in raw_response:
            return ([], "Your query seems unclear or incomplete. Could you please clarify?")

        valid_intents = {"get_data_from_tables", "bar_chart", "line_chart", "pie_chart", "column_chart", "get_data_from_history"}
        extracted = re.findall(r'\b(?:get_data_from_tables|bar_chart|line_chart|pie_chart|column_chart|get_data_from_history)\b', raw_response)

        final_intents = list(set(extracted).intersection(valid_intents)) or ["get_data_from_tables"]

        return (final_intents, "")
    

    def extract_data_from_history(self, query: str, memory):
        
        chat_history = self.summarize_messages(memory.chat_memory.messages, max_messages=10) if memory else "No history."
        chat_history = self.clean_chat_history(chat_history)

        prompt_template = PromptTemplate(
            input_variables=["user_query", "chat_history"],
            template="""
                You are an AI assistant tasked with extracting structured JSON data from chat history.
                Chat history (cleaned for your reference):
                {chat_history}
                User query:
                {user_query}
                Your task:
                - Search the chat history for relevant information based on the user's query.
                - Return the information in the following JSON format don't add any extra space, quotes just a clean JSON:
                [
                    {{
                        "key": "<key>",
                        "value": "<value>"
                    }},
                    ...
                ]
                If no relevant information is found, return:
                    []
                Ensure the JSON is properly formatted and valid.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        raw_response = chain.run({
            "user_query": query,
            "chat_history": chat_history
        }).strip()

        try:
            structured_data = json.loads(raw_response)
        except json.JSONDecodeError:
            logging.error(f"Malformed JSON response: {raw_response}")
            structured_data = []

        return structured_data

    def clean_chat_history(self, chat_history: str) -> str:
        chat_history = re.sub(r"<[^>]*>", "", chat_history)
        chat_history = re.sub(r"\s+", " ", chat_history).strip()
        return chat_history[:5000]  # Limit to 5000
    

    def _get_chart_template(self, chart_type: str) -> str:
        """Get specific template instructions based on chart type."""
        templates = {
            "bar_chart": """
                For bar charts:
                - X-axis should contain categorical data
                - Y-axis must contain numerical values
                - Sort bars by y-axis values if not time-based
                - Return data ordered by y-values in descending order for better visualization
                - Limit to top 15 categories if there are too many
            """,
            "line_chart": """
                For line charts:
                - X-axis should preferably be temporal or sequential
                - Y-axis must contain numerical values
                - Sort data points by x-axis values
                - Ensure x-axis values are properly ordered
                - If dates are present, format them consistently
            """,
            "pie_chart": """
                For pie charts:
                - X-axis should contain categories (labels)
                - Y-axis must contain positive numerical values
                - Consider grouping small values into "Other" category
                - Limit to top 8 categories, grouping remainder into "Other"
                - Sort values in descending order
            """
        }
        return templates.get(chart_type, "")

    def prepare_data_for_chart(self, json_data: list, chart_type: str, user_query: str):
        def convert_decimal(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        prompt_template = PromptTemplate(
            input_variables=["json_data", "chart_type", "user_query"],
            template="""
                You are an AI assistant that transforms raw JSON data into structured rows for generating charts using matplotlib.
                The JSON data provided is:
                {json_data}
                The user query is:
                {user_query}
                The user wants to generate a {chart_type}. Your task is:
                1) Identify the best columns for the x-axis and y-axis based on the chart type and user query.
                2) Ensure that the x-axis contains labels or sequential values, and the y-axis contains numerical values.
                3) Ignore rows with missing, null, or invalid data for the selected x and y columns.
                4) Return ONLY a list of dictionaries with this EXACT structure (do not include any other text):
                [
                    {{"x_col": <x_value>, "y_col": <y_value>}},
                    ...
                ]
            """
        )

        try:
            json_data_serialized = json.dumps(json_data, indent=2, default=convert_decimal)

            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = chain.run({
                "json_data": json_data_serialized,
                "chart_type": chart_type,
                "user_query": user_query
            }).strip()

            response = response.replace('```json', '').replace('```', '').strip()

            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > 0:
                json_str = response[start_idx:end_idx]

                parsed_data = json.loads(json_str)
                if not isinstance(parsed_data, list):
                    return []

                formatted_data = []
                for item in parsed_data:
                    if isinstance(item, dict):
                        formatted_item = {
                            "x_col": item.get('"x_col"') or item.get('x_col'),
                            "y_col": item.get('"y_col"') or item.get('y_col')
                        }
                        if formatted_item["x_col"] is not None and formatted_item["y_col"] is not None:
                            formatted_data.append(formatted_item)

                return formatted_data

            return []

        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return []

    def generate_final_answer(self, user_query: str, sql_results: list, intents: list, generated_charts: dict = None, generate_title: bool = True, error_message: str = None):
        if error_message:
            results_summary = ""
            chart_descriptions = ""
        else:
            results_summary = QueryService.summarize_results(sql_results)
            chart_descriptions = "\n".join(
                f"{chart_type.replace('_', ' ').title()}: {url}" for chart_type, url in (generated_charts or {}).items()
            )
        
        merged_input = f"""
        User's Query: {user_query}
        {"Error Message: " + error_message if error_message else f"Summary: {results_summary}"}
        User Intents: {", ".join(intents)}
        Generated Charts (URLs): 
        {chart_descriptions}
        Generate Title: {"true" if generate_title else "false"}
        """
        
        input_tokens = Helper.count_tokens(merged_input, model="gpt-4o-mini")
        max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
        self.llm.max_tokens = min(max_completion_tokens, 16384)
        
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="""
            You are an AI assistant tasked with generating a well-structured HTML response, and optionally, a title.
            Input Information:
            {input}
            Your task:
            - Use the provided information to create a complete and user-friendly HTML response. 
            - Don't add <h1>, <h2>, <html>, <body>, header, or footer tags, and do not include CSS in the response.
            - Include the results summary as a section in the HTML if no error message is provided.
            - Embed the provided chart URLs as <img> tags under appropriate headings, if available.
            - If "Generate Title" is true, create a concise and unique title based only on the user's query or error message.
            Otherwise, set the title to an empty string.
            - If an error message is provided, ignore the results summary and charts, and base the response solely on the error message.
            Return your response in the following format:
            Title: <generated title or "">
            HTML: <HTML response, don't add any quotes or extra spaces just a clean HTML>
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run({"input": merged_input}).strip()
        try:
            title_start = response.index("Title:") + len("Title:")
            html_start = response.index("HTML:")
            title = response[title_start:html_start].strip()
            html = response[html_start + len("HTML:"):].strip()
        except ValueError:
            title = ""
            html = response

        return html, title



    def generate_contextual_answer(self, query: str, context: str, generate_title: bool, memory=None):
        chat_history_string = self.summarize_messages(memory.chat_memory.messages)
        combined_context = self.get_relevant_context(context, max_contexts=3)

        generate_title_text = "True" if generate_title else "False"
        merged_input = (
            f"User's Query: {query}\n"
            f"Context: {combined_context}\n"
            f"Generate Title: {generate_title_text}\n"
            f"Chat History: {chat_history_string}"
        )

        input_tokens = Helper.count_tokens(merged_input, model="gpt-4o-mini")
        max_completion_tokens = self.calculate_available_tokens(128000, input_tokens)
        self.llm.max_tokens = min(max_completion_tokens, 16384)

        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="""
            You are an AI assistant.
            Below is the user's question, context, and chat history (memory).
            {input}
            Guidelines:
            - Use only both the provided context and the history (memory) of the conversation to provide a well-informed and accurate response.
            - If the user's query is unclear but the context or history is sufficient, respond politely asking for clarification on the query while leveraging the available information to assist as much as possible.
            - If both the query and the context are unclear, and no helpful information is available in the history, respond politely asking the user to provide more details without making assumptions or generating speculative content.
            - Ensure that the answer is complete and do not leave sentences or points unfinished.
            - If the answer cannot be fully generated due to token limits, summarize the key points instead of truncating.
            - Always return the answer in HTML format and the Title in TEXT format.
            - Don't add <h1>, <h2>, <html>, <body>, header, or footer tags, and do not include CSS in the HTML answer.
            - Incorporate relevant insights or information from the history (memory) to enhance the response whenever applicable.
            Generate the following in a single response:
            - Answer: Provide a response in HTML format based on the guidelines.
            - Title: Create a concise and unique title if `generate_title` is true based on user query and don't look into context; otherwise, return "".
            Return your response in the following format:
            Answer: <answer in valid HTML format, don't add any quotes or extra spaces>
            Title: <title text or "">
            Remember strictly to not give any information from the internet, only use context, memory, and query.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt_template, memory=memory)

        try:
            response = chain({"input": merged_input})
        except Exception as e:
            if "context length exceeded" in str(e):
                return (
                    "<p>Error: Input too large. Please refine your question.</p>",
                    "",
                    "",
                )
            raise e

        response_text = response["text"].strip()
        answer_start = response_text.find("Answer:") + len("Answer:")
        title_start = response_text.find("Title:")
        answer = response_text[answer_start:title_start].strip()
        title = response_text[title_start + len("Title:"):].strip()

        return answer, title, response_text

    @staticmethod
    def calculate_available_tokens(total_limit, input_tokens):
        return max(total_limit - input_tokens, 0)

    @staticmethod
    def summarize_messages(messages, max_length=1000, max_messages = 10):
        if not messages:
            return "No history available."
        
        summary = ""
        for msg in messages[-max_messages:]:
            if isinstance(msg, HumanMessage):
                summary += f"Human: {msg.content[:max_length]}...\n"
            elif isinstance(msg, AIMessage):
                summary += f"Assistant: {msg.content[:max_length]}...\n"
        return summary.strip()


    @staticmethod
    def get_relevant_context(context, max_contexts=3):
        context_list = context.split(". ") 
        return ". ".join(context_list[:max_contexts]) 
    

    def generate_sql_query(self, user_query: str, schema_description: str, table_names: list, database: str, rights_schema: str, user_email: str, rights_table: str, distinct_data: list):
        user_info_str = f"user email = {user_email}"
        
        # Format the latest distinct CTC records for inclusion in the prompt
        ctc_data_str = "\n".join([
            f"Company: {record['COMPANY_NAME']} (Code: {record['COMPANY_CODE']}), "
            f"Project: {record['PROJECT_NAME']} (Code: {record['PROJECT_CODE']}), "
            f"CTC Status: {record['CTC_STATUS']}, "
            f"Total Budget: {record['TOTAL_BUDGET']}, Total Cost at Completion: {record['TOTAL_COST_AT_COMPLETION']}, "
            f"Variation at Completion: {record['VARIATION_AT_COMPLETION']}"
            for record in distinct_data
        ])
        
        prompt_template = PromptTemplate(
            input_variables=["user_query", "schema_description", "table_names", "database", "rights_schema", "user_info", "rights_table", "latest_ctc_data"],
            template="""
            You are an AI assistant that converts natural language queries into SQL Server SELECT statements.
            The database is '{database}'.
            The following tables/views are available: {table_names}, rights table: {rights_table}
            Schemas:
            {schema_description}
            Rights Schema:
            {rights_schema}
            User Information:
            {user_info}
            Latest CTC Records Example (distinct by company):
            The following records are the latest Cost-to-Complete (CTC) data, with one record per company:
            {latest_ctc_data}
            Requirements:
            - Always return valid SQL Server SELECT statement only with minimum number of columns.
            - Use WHERE with LIKE '%...%' instead of '=' for string comparisons.
            - No extra explanations or code fences.
            - Enforce user rights as described.
            - Just generate SELECT sql query and don't put anything else in the answer.
            User query: "{user_query}"
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(
            user_query=user_query,
            schema_description=schema_description,
            table_names=", ".join(table_names),
            database=database,
            rights_schema=rights_schema,
            user_info=user_info_str,
            rights_table=rights_table,
            latest_ctc_data=ctc_data_str
        )
        return response.strip()


