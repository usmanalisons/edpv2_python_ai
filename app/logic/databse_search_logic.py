import re
import xml.etree.ElementTree as ET
from app.services.sql_db_service import SQLDatabaseService
from app.services.database_search_prompt_service import DatabaseSearchPromptService
from app.core.config import settings
from app.services.chat_memory_service import ChatMemoryService
from uuid import uuid4
from app.utils.helper import Helper
import json
import asyncio


class DatabaseSearchLogic:
    def __init__(self, chat_memory_service):
        self.chat_memory_service: ChatMemoryService = chat_memory_service
        self.intranet_db_service = SQLDatabaseService(connection_name="intranet")
        self.ctc_db_service = SQLDatabaseService(connection_name="ctc")
        self.db_prompt_service = DatabaseSearchPromptService(model_name="gpt-4o", api_key=settings.OPENAI_API_KEY)

    def sanitize_sql(self, sql_query: str) -> str:
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        return sql_query

    def build_xml_schema(self, tables: list) -> str:
        root = ET.Element("Database")
        for table_name in tables:
            schema_results = self.ctc_db_service.get_table_schema(table_name)
            table_element = ET.SubElement(root, "Table", name=table_name)
            for row in schema_results:
                col_name = row.get('COLUMN_NAME') or 'UNKNOWN'
                data_type = row.get('DATA_TYPE') or 'UNKNOWN'
                column_comment = row.get('COLUMN_COMMENT') or ''
                ET.SubElement(table_element, "Column", name=col_name, type=data_type, description=column_comment)
        return ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")

    def build_rights_schema(self, rights_table: str) -> str:
        rights_schema_results = self.ctc_db_service.get_table_schema(rights_table)
        if not rights_schema_results:
            return "No rights schema available."
        
        columns_info = []
        for row in rights_schema_results:
            col_name = row.get('COLUMN_NAME', 'UNKNOWN')
            data_type = row.get('DATA_TYPE', 'UNKNOWN')
            data_comment = row.get('COLUMN_COMMENT', '')
            columns_info.append(f"{col_name} ({data_type}) ({data_comment})")
        return f"Columns in {rights_table}:\n" + ", ".join(columns_info)

    async def sql_execution_with_regenerate(
        self,
        user_query: str,
        schema_info: dict,
        user_email: str,
        ctc_data_by_companies,
        max_retries: int = 3
    ):
        last_error = None
        current_user_query = user_query

        for attempt in range(max_retries):
            # sql_query = self.db_prompt_service.generate_sql(
            #     user_question=current_user_query,
            #     schema_description=tables_schema,
            #     table_names=tables,
            #     database=db_name,
            #     rights_schema=rights_schema,
            #     user_email=user_email,
            #     rights_table=rights_table,
            #     distinct_data=ctc_data_by_companies
            # )

            sql_query = await self.db_prompt_service.generate_sql(
                    question=current_user_query,
                    schema_info=schema_info,
                    user_email=user_email,
                    sample_data=ctc_data_by_companies
            )
            sql_query = self.sanitize_sql(sql_query)
            try:
                rows = self.ctc_db_service.run_query(sql_query)
                return rows, sql_query, None
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    current_user_query = (
                        f"{user_query}\n"
                        f"SQL_ERROR: {last_error}\n"
                        "Please fix the SQL query based on the above error and regenerate."
                    )
                    # Adding 0.5-second sleep before retrying
                    await asyncio.sleep(0.5)
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
#         rows = [
#     {
#         "RISK_CATEGORY": None,
#         "PROJECT_COUNT": 56
#     },
#     {
#         "RISK_CATEGORY": "High Risk",
#         "PROJECT_COUNT": 20
#     },
#     {
#         "RISK_CATEGORY": "Low Risk",
#         "PROJECT_COUNT": 5
#     },
#     {
#         "RISK_CATEGORY": "Medium Risk",
#         "PROJECT_COUNT": 6
#     },
#     {
#         "RISK_CATEGORY": "No Risk",
#         "PROJECT_COUNT": 52
#     }
# ]
        
#         refined_query = "Generate a bar chart and pie chart showing the distribution of projects under different RISK_CATEGORY for the company with COMPANY_CODE 'asmef'."
#         chart_types = ['bar_chart', 'pie_chart']

#         chart_data = await self.db_prompt_service.prepare_chart_data(rows, refined_query, chart_types)

#         return chart_data

        chat = None
        is_new_chat = False
        chat_memory = None
        final_sql_query = ""
    
        if not chat_id:
            is_new_chat = True
            chat_id = str(uuid4())
            chat_memory = await self.chat_memory_service.create_memory(chat_id, [])
        else:
            chat = self.intranet_db_service.get_chat_by_id(chat_id)
            if not chat:
                return None
            if await self.chat_memory_service.has_memory(chat_id):
                chat_memory = await self.chat_memory_service.get_memory(chat_id)
            else:
                previous_messages = self.intranet_db_service.get_chat_messages(chat_id)
                chat_memory = await self.chat_memory_service.create_memory(chat_id, previous_messages)

            print(f"Memmory Messages: ", chat_memory.chat_memory.messages)

        tables = [
            "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE",
            "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE_BY_MONTH",
            "VIEW_COMPANY_PROJECT_COST_TO_COMPLETE_BY_CATEGORY",
            "VIEW_REVENUE_DETAILS"
        ]
        db_name = "ctc_db.dbo"
        rights_table = "VIEW_COMPANY_RIGHTS"

        xml_schema = self.build_xml_schema(tables)
        rights_schema = self.build_rights_schema(rights_table)

        # print('xml_schema: ', xml_schema)
        # print('rights_schema: ', rights_schema)
        # print('query: ', query)

        ctc_data_by_companies = self.ctc_db_service.get_top_distinct_ctc_data()

        analyze_response = await self.db_prompt_service.analyze_question(query,  xml_schema, memory=chat_memory, sample_data=ctc_data_by_companies)
        return analyze_response

        intents = analyze_response.get('intents', [])
        error = analyze_response.get('error', '')
        refined_query = analyze_response.get('refined_query', '')
        chart_types = analyze_response.get('chart_types', [])


        print(f'intents: {intents}')
        print(f'refined_query: {refined_query}')
        print(f'chart_types: {chart_types}')
        print(f'error_message: {error}')

        if error:
            await asyncio.sleep(0.5)
            return await self.process_ctc_response(
                query=query, 
                refined_query=refined_query, 
                intents=intents, 
                final_sql_query=
                final_sql_query, 
                rows=[], generated_charts=[], 
                chat_id = chat_id, 
                user_email = user_email, 
                is_new_chat = is_new_chat, error_message = error
            )
        
        rows = []

        ctc_data_by_companies = self.ctc_db_service.get_top_distinct_ctc_data()
       

        #  "database": schema_info["database"],
        #     "table_names": schema_info["tables"],
        #     "schema_description": schema_info["schema"],
        #     "rights_schema": schema_info["rights_schema"],
        #     "rights_table": schema_info["rights_table"],
        

        schema_info = {
            "database": db_name,
            "tables":  tables,
            "schema":  xml_schema,
            "rights_schema":  rights_schema,
            "rights_table":  rights_table,

        }

        await asyncio.sleep(0.5)
        rows, final_sql_query, error_msg = await self.sql_execution_with_regenerate(
            user_query=refined_query,
            schema_info=schema_info,
            user_email=user_email,
            ctc_data_by_companies = ctc_data_by_companies,
            max_retries=3
        )

        if error_msg:
            error = "Could not get data, please report back to IT support team."
            # return await self.process_ctc_response(
            #     query=query, 
            #     refined_query=refined_query, 
            #     final_sql_query=final_sql_query, 
            #     rows=[], 
            #     generated_charts=[], 
            #     chat_id = chat_id, 
            #     user_email = user_email, 
            #     is_new_chat = is_new_chat, 
            #     error_message = error
            # )
        
            return await self.process_ctc_response(
                query=query, 
                refined_query=refined_query, 
                final_sql_query=final_sql_query, 
                chat_id = chat_id, 
                user_email = user_email, 
                is_new_chat = is_new_chat,
                error_message = error
            )

        generated_charts = {}
        # if rows and chart_types:
        #     charts_data_list = await self.db_prompt_service.prepare_chart_data(rows, refined_query, chart_types)
        #     return charts_data_list
        #         chart_path = Helper.generate_chart(chart_data, chart_type=chart_type, output_dir="app/charts", chart_name=chart_type)
        #         public_url = chart_path.replace("app/charts", 'http://10.100.55.23:8001/charts').replace("\\", "/")
        #         generated_charts[chart_type] = public_url



        charts_data_obj = {}
        if rows and chart_types:
            charts_data_obj = await self.db_prompt_service.prepare_chart_data(rows, refined_query, chart_types)
            for chart_type, chart_data in charts_data_obj.items():

                if chart_data:
                    chart_path = Helper.generate_chart(
                        chart_data=chart_data,
                        chart_type=chart_type,
                        output_dir="app/charts",
                        chart_name=chart_type
                    )
                    if chart_path and not chart_path.startswith("Could not"):  # Check for valid path
                        public_url = chart_path.replace("app/charts", 'http://10.100.55.23:8001/charts').replace("\\", "/")
                        generated_charts[chart_type] = public_url

       
        return await self.process_ctc_response(
            query=query, 
            refined_query=refined_query, 
            final_sql_query=final_sql_query, 
            rows=rows, 
            generated_charts=generated_charts, 
            chat_id = chat_id, 
            user_email = user_email, 
            is_new_chat = is_new_chat
        )

    async def process_ctc_response(
        self,
        query: str,
        refined_query: str,
        final_sql_query: str,
        rows: list,
        generated_charts: dict,
        chat_id: str,
        user_email: str,
        is_new_chat: bool,
        error_message: str = None
    ):
        print(final_sql_query)
        response = await self.db_prompt_service.format_response(
            question=refined_query,
            sql_results=rows,
            charts=generated_charts,
            error_message=error_message,
            generate_title=True
        )

        print(response)
    
        title = response.get('title', "")
        text = response.get('text', "")
        html = response.get('html', "")

        # return response

        await self.chat_memory_service.add_assistant_message(chat_id, query)
        self.intranet_db_service.add_chat_message(chat_id, 'HUMAN', query)

        chat = None
        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, 'CTC', user_email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, html)


        # data_converted = Helper.convert_decimals_in_obj(rows)
        # data_str = json.dumps(data_converted)
                
        self.intranet_db_service.add_chat_message(chat_id, 'ASSISTANT', html, html, "", final_sql_query)

        return {
            "data": {
                "type": "text",
                "charts": generated_charts,
                "answer": html,
                "sql_query": final_sql_query,
                "sql_results": rows,
                "title": title,
                "chat": chat,
            }
        }

