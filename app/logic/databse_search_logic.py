import re
import xml.etree.ElementTree as ET
from app.services.sql_db_service import SQLDatabaseService
from app.services.query_service import QueryService
from app.core.config import settings
import logging
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain_openai.chat_models import ChatOpenAI
from app.services.chat_memory_manager_service import ChatMemoryManager
from uuid import uuid4
from app.utils.helper import Helper

class DatabaseSearchLogic:
    def __init__(self):
        self.query_service = QueryService(model_name="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
        self.logger = logging.getLogger(__name__)
        

    def sanitize_sql(self, sql_query: str) -> str:
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        return sql_query

    def build_xml_schema(self, sql_service: SQLDatabaseService, tables: list) -> str:
        root = ET.Element("Database")
        for table_name in tables:
            schema_results = sql_service.get_table_schema(table_name)
            table_element = ET.SubElement(root, "Table", name=table_name)
            for row in schema_results:
                col_name = row.get('COLUMN_NAME') or 'UNKNOWN'
                data_type = row.get('DATA_TYPE') or 'UNKNOWN'
                column_comment = row.get('COLUMN_COMMENT') or ''
                ET.SubElement(table_element, "Column", name=col_name, type=data_type, description=column_comment)
        return ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")

    def build_rights_schema(self, sql_service: SQLDatabaseService, rights_table: str) -> str:
        rights_schema_results = sql_service.get_table_schema(rights_table)
        if not rights_schema_results:
            return "No rights schema available."
        
        columns_info = []
        for row in rights_schema_results:
            col_name = row.get('COLUMN_NAME', 'UNKNOWN')
            data_type = row.get('DATA_TYPE', 'UNKNOWN')
            data_comment = row.get('COLUMN_COMMENT', '')
            columns_info.append(f"{col_name} ({data_type}) ({data_comment})")
        return f"Columns in {rights_table}:\n" + ", ".join(columns_info)

    def sql_execution_with_regenerate(
        self,
        user_query: str,
        tables_schema: str,
        tables: list,
        db_name: str,
        rights_schema: str,
        user_email: str,
        rights_table: str,
        sql_service: SQLDatabaseService,
        ctc_data_by_companies,
        max_retries: int = 3
    ):
        last_error = None
        current_user_query = user_query

        for attempt in range(max_retries):
            sql_query = self.query_service.generate_sql_query(
                user_query=current_user_query,
                schema_description=tables_schema,
                table_names=tables,
                database=db_name,
                rights_schema=rights_schema,
                user_email=user_email,
                rights_table=rights_table,
                distinct_data=ctc_data_by_companies
            )
            sql_query = self.sanitize_sql(sql_query)
            try:
                rows = sql_service.run_query(sql_query)
                return rows, sql_query, None
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    current_user_query = (
                        f"{user_query}\n"
                        f"SQL_ERROR: {last_error}\n"
                        "Please fix the SQL query based on the above error and regenerate."
                    )
                else:
                    return [], sql_query, last_error

        return [], "", "Unexpected loop exit."
    

    def map_intent_to_chart_type(self, intent: str):
        if intent in {"bar_chart", "line_chart", "pie_chart", "column_chart"}:
            return intent
        if intent == "trend":
            return "line_chart"
        if intent == "visualize":
            return "bar_chart"
        return None

    async def handle_ctc_search(self, query: str, user_email: str, chat_id: str):
        chat = None
        is_new_chat = False
        chat_memory = None
    
        intranet_db_service = SQLDatabaseService(connection_name="intranet")
        memory_manager = ChatMemoryManager(openai_api_key=settings.OPENAI_API_KEY)

        if not chat_id:
            is_new_chat = True
            chat_id = str(uuid4())
            chat_memory = memory_manager.create_memory(chat_id, [])
        else:
            chat = intranet_db_service.get_chat_by_id(chat_id)
            if not chat:
                return None
            if memory_manager.has_memory(chat_id):
                chat_memory = memory_manager.get_memory(chat_id)
            else:
                previous_messages = intranet_db_service.get_chat_messages(chat_id)
                chat_memory = memory_manager.create_memory(chat_id, previous_messages)

            print(f"Memmory Messages: ", chat_memory.chat_memory.messages)

        tables = [
            "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE",
            "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE_BY_MONTH",
            "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE_BY_CATEGORY"
        ]
        db_name = "ctc_db.dbo"
        rights_table = "VIEW_COMPANY_RIGHTS"

        sql_service = SQLDatabaseService(connection_name="ctc")

        xml_schema = self.build_xml_schema(sql_service, tables)
        rights_schema = self.build_rights_schema(sql_service, rights_table)

        intents, clarifying_msg = self.query_service.analyze_query_intent(query, xml_schema, memory=chat_memory)

        print(f'intents: {intents}')
        print(f'clarifying_msg: {clarifying_msg}')

        if clarifying_msg:
            return await self.process_ctc_response(query, [], intents, [], chat_id, user_email, is_new_chat, memory_manager, intranet_db_service, clarifying_msg)
    
        error_msg = ""
        rows = []
        if "get_data_from_history" in intents:
            print(f'GETTING DATA FROM HISTORY')
            rows = self.query_service.extract_data_from_history(query, memory=chat_memory)
        else:
            print(f'GETTING DATA FROM SQL')

            ctc_data_by_companies = sql_service.get_top_distinct_ctc_data()
            rows, final_sql_query, error_msg = self.sql_execution_with_regenerate(
                user_query=query,
                tables_schema=xml_schema,
                tables=tables,
                db_name=db_name,
                rights_schema=rights_schema,
                user_email=user_email,
                rights_table=rights_table,
                sql_service=sql_service,
                ctc_data_by_companies = ctc_data_by_companies,
                max_retries=3
            )

            if error_msg:
                error = "Could not get data"
                return await self.process_ctc_response(query, [], intents, [], chat_id, user_email, is_new_chat, memory_manager, intranet_db_service, error)
            
        generated_charts = {}

        if rows and intents:
            for intent in intents:
                chart_type = self.map_intent_to_chart_type(intent)
                if not chart_type:
                    continue
                if chart_type:
                    response = self.query_service.prepare_data_for_chart(rows, chart_type, user_query = query)

                    # return response
                    chart_path = Helper.plot_chart(response, chart_type=chart_type, output_dir="app/charts", chart_name_prefix=intent)
                    public_url = chart_path.replace("app/charts", 'http://localhost:8001/charts').replace("\\", "/")
                    generated_charts[intent] = public_url

       
        return await self.process_ctc_response(query, rows, intents, generated_charts, chat_id, user_email, is_new_chat, memory_manager, intranet_db_service)

    async def process_ctc_response(
        self,
        query: str,
        rows: list,
        intents: list,
        generated_charts: dict,
        chat_id: str,
        user_email: str,
        is_new_chat: bool,
        memory_manager: ChatMemoryManager,
        intranet_db_service: SQLDatabaseService,
        error_message: str = None
    ):
        html, title = self.query_service.generate_final_answer(
            user_query=query,
            sql_results=rows,
            intents=intents,
            generated_charts=generated_charts,
            error_message=error_message
        )

        memory_manager.add_user_message(chat_id, query)
        intranet_db_service.add_chat_message(chat_id, 'HUMAN', query)

        chat = None
        if is_new_chat:
            chat = intranet_db_service.add_chat(chat_id, 'CTC', user_email, title)

        memory_manager.add_assistant_message(chat_id, html)
        intranet_db_service.add_chat_message(chat_id, 'ASSISTANT', html)

        return {
            "data": {
                "type": "text",
                "charts": generated_charts,
                "answer": html,
                "sql_query": "",
                "sql_results": rows,
                "title": 'CTC_SEARCH',
                "chat": chat,
            }
        }

