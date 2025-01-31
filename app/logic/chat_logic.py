import os
import json
import re
import asyncio
import xml.etree.ElementTree as ET
from uuid import uuid4
from fastapi import UploadFile
from app.utils.file_loader import load_text_from_file
from app.services.text_processor_service import TextProcessorService
from app.services.document_search_prompt_service import DocumentSearchPromptService
from app.services.database_search_prompt_service import DatabaseSearchPromptService
from app.services.chat_prompt_service import ChatPromptService
from app.utils.helper import Helper
from app.core.config import settings
from app.core.constants import MAX_TOKENS_PER_REQUEST
from app.services.sql_db_service import SQLDatabaseService
from app.services.chat_memory_service import ChatMemoryService
from app.services.chroma_db_service import ChromaDBService
from app.models.request_models import SendChatRequestUser
from langchain.memory import ConversationBufferMemory
import pandas as pd
from typing import List, Dict


class ChatLogic:
    def __init__(self, chroma_db_service, chat_memory_service):
        self.chroma_db_service: ChromaDBService = chroma_db_service
        self.chat_memory_service: ChatMemoryService = chat_memory_service
        self.intranet_db_service = SQLDatabaseService(connection_name="intranet")
        self.ctc_db_service = SQLDatabaseService(connection_name="ctc")
        self.quantum_db_service = SQLDatabaseService(connection_name="quantum")
        self.text_processor_service = TextProcessorService(model_name=settings.OPENAI_EMBEDDING_MODEL)
        self.chat_prompt_service = ChatPromptService(model_name="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
        self.document_prompt_service = DocumentSearchPromptService(model_name="gpt-4o", api_key=settings.OPENAI_API_KEY)
        self.db_prompt_service = DatabaseSearchPromptService(model_name="gpt-4o", api_key=settings.OPENAI_API_KEY)
        self.db_mini_prompt_service = DatabaseSearchPromptService(model_name="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

    def get_employee_chats(self, email, type):
        chats = self.intranet_db_service.get_employee_chats(email, type)
        return {
            "data": chats
        }
    
    def get_chat_details(self, chat_id):
        chat = self.intranet_db_service.get_chat_by_id(chat_id)
        if chat:
            messages = self.intranet_db_service.get_chat_messages(chat_id)
            chat['messages'] = messages
        return {
            "data": chat
        }
    
    def get_chat_messages(self, chat_id):
        messages = self.intranet_db_service.get_chat_messages(chat_id)
        return {
            "data": messages
        }
    

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
        sample_ctc_data,
        max_retries: int = 3
    ):
        last_error = None
        current_user_query = user_query

        for attempt in range(max_retries):
            sql_query = await self.db_prompt_service.generate_sql(
                    question=current_user_query,
                    schema_info=schema_info,
                    user_email=user_email,
                    sample_data=sample_ctc_data
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


    async def send_chat_message(self, question: str, user: SendChatRequestUser, chat_id: str = None,):
        collection_key = "policiesProcedures"
        chat_memory = None
        is_new_chat = False

        chat = None
        if not chat_id:
            is_new_chat = True
            chat_id = str(uuid4())
            chat_memory = await self.chat_memory_service.create_memory(chat_id, [])
        else:
            chat = self.intranet_db_service.get_chat_by_id(chat_id)
            if not chat:
                return {"message": "Chat ID not found."}

            if await self.chat_memory_service.has_memory(chat_id):
                chat_memory = await self.chat_memory_service.get_memory(chat_id)
            else:
                previous_messages = self.intranet_db_service.get_chat_messages(chat_id)
                chat_memory = await self.chat_memory_service.create_memory(chat_id, previous_messages)

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

        ctc_data_by_companies = self.ctc_db_service.get_top_distinct_ctc_data()
        analyze_response = await self.chat_prompt_service.analyze_question(question, schema_info=xml_schema, memory=chat_memory, sample_data=ctc_data_by_companies)

        print(analyze_response.classification)
        print(analyze_response.refined_question)
        print(analyze_response.chart_types)
        print(analyze_response.error)

        # return analyze_response

        classification = analyze_response.classification
        error = analyze_response.error
        refined_question = analyze_response.refined_question
        chart_types = analyze_response.chart_types


        # classification = "CTC"
        # refined_question = "Could you provide the monthly revenue trend for each project in 2024 across all companies, including a comparison of the revenue between different projects?"
        # chart_types = ['line_chart']
        # error = "The question is somewhat clear, but specifying the exact risk categories (High Risk, Medium Risk, Low Risk, No Risk) would help in providing a more focused response."

        if classification == 'NONE':
            return await self.get_error_answer(question, refined_question, user, chat_id,chat, is_new_chat, chat_memory, error)

        if(classification == 'POLICIES'):
            return await self.get_policies_final_answer(question, refined_question, user, chat_id, chat, is_new_chat, chat_memory)
        elif(classification == 'ORACLE'):
            return await self.get_oracle_trainings_final_answer(question, refined_question, user, chat_id, chat, is_new_chat, chat_memory)
        elif(classification == 'CTC'):
            return await self.get_database_final_answer(question, refined_question, user, chat_id, chat, is_new_chat, chat_memory, chart_types, db_name, tables, xml_schema, rights_schema, rights_table, ctc_data_by_companies)
        else:
            return await self.get_error_answer(question, refined_question, user, chat_id,chat, is_new_chat, chat_memory, error)
        

    async def get_policies_final_answer(self, question: str, refined_question: str, user: SendChatRequestUser, chat_id: str, chat, is_new_chat: bool, chat_memory: ConversationBufferMemory):
        collection_key = "policiesProcedures"
        filters = Helper.filter_policies_procedures(user.email, user.department_code, user.company_code)
        retriever = await self.chroma_db_service.retrieve_as_retriever(
            collection_key=collection_key,
            filter_dict=filters
        )
        relevant_docs = retriever.invoke(
            input=refined_question,
        )
        contexts = [doc.page_content for doc in relevant_docs]
        combined_context = " ".join(contexts)
        response = await self.document_prompt_service.generate_answer(question, combined_context, chat_memory, "Policies & Procedures", is_new_chat)

        text_answer = response.get('text_answer', '')
        html_answer = response.get('html_answer', '')
        context_based = response.get('context_based', '')
        title = response.get('title', [])

        await self.chat_memory_service.add_human_message(chat_id, question)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", question)

        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, "POLICIES_PROCEDURES", user.email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, text_answer)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", text_answer, html_answer)

        return {
            "data": {
                "type": "DOCUMENT",
                "title": title,
                "answer": html_answer,
                "chat_id": chat_id,
                "chat": chat,
                "relevant_docs": relevant_docs,
                "combined_context": combined_context,
            }
        }
    
    async def get_oracle_trainings_final_answer(self, question: str, refined_question: str, user: SendChatRequestUser, chat_id: str, chat, is_new_chat: bool, chat_memory: ConversationBufferMemory):
        collection_key = "oracleTrainings"
        retriever = await self.chroma_db_service.retrieve_as_retriever(
            collection_key=collection_key,
        )
        relevant_docs = retriever.invoke(
            input=refined_question,
        )
        contexts = [doc.page_content for doc in relevant_docs]
        combined_context = " ".join(contexts)
        response = await self.document_prompt_service.generate_answer(question, combined_context, chat_memory, "Oracle Trainings Wave 2", is_new_chat)


        text_answer = response.get('text_answer', '')
        html_answer = response.get('html_answer', '')
        context_based = response.get('context_based', '')
        title = response.get('title', [])

        await self.chat_memory_service.add_human_message(chat_id, question)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", question)

        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, "ORACLE_TRAININGS", user.email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, text_answer)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", text_answer, html_answer)

        return {
            "data": {
                "type": "DOCUMENT",
                "title": title,
                "answer": html_answer,
                "chat_id": chat_id,
                "chat": chat,
                "relevant_docs": relevant_docs,
                "combined_context": combined_context,
            }
        }

    def summarize_large_data_dynamic(self, rows):
        df = pd.DataFrame(rows)
        
        summary = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary[col] = {
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "sum": df[col].sum()
                }
            else:
                summary[col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
        
        return summary
    

    def _format_data_summary(self, data: List[Dict], summarize: bool = False) -> str:
        if not data:
            return "No data available."
        df = pd.DataFrame(data)
        if summarize:
            summary = []
            
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary.append(f"{col} - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Sum: {df[col].sum()}")
                else:
                    summary.append(f"{col} - Unique Values: {df[col].nunique()}, Top Values: {df[col].value_counts().head(5).to_dict()}")
            
            return "\n".join(summary)
        columns = list(df.columns)
        summary = ["Columns: " + ", ".join(columns)]
        
        for idx, row in enumerate(data, start=1):
            row_data = ", ".join([f"{key}: {value}" for key, value in row.items()])
            summary.append(f"Row {idx}: {row_data}")
        
        return "\n".join(summary)

    async def get_database_final_answer(
       self, 
       question: str, 
       refined_question: str, 
       user: SendChatRequestUser, 
       chat_id: str, 
       chat,
       is_new_chat: bool, 
       chat_memory: ConversationBufferMemory, 
       chart_types, 
       db_name: str, 
       tables, 
       tables_schema, 
       rights_schema, 
       rights_table, 
       sample_ctc_data,
    ):
        rows = []
        schema_info = {
            "database": db_name,
            "tables":  tables,
            "schema":  tables_schema,
            "rights_schema":  rights_schema,
            "rights_table":  rights_table,

        }

        await asyncio.sleep(0.5)
        error_message = ""
        final_sql_query=""
        rows, final_sql_query, error_message = await self.sql_execution_with_regenerate(
            user_query=refined_question,
            schema_info=schema_info,
            user_email=user.email,
            sample_ctc_data = sample_ctc_data,
            max_retries=3
        )

        # file_location = f"app/sample_data.json"
        # with open(file_location, 'r') as file:
        #     rows = json.load(file)


        MAX_ROWS = 300

        # if rows and len(rows) > MAX_ROWS:
        #     rows = self.summarize_large_data_dynamic(rows)
        # else:
        #     rows = rows

        rows_summary = self._format_data_summary(rows, rows and len(rows) > MAX_ROWS)

        if error_message:
            error_message = "Could not get data, please report back to IT support team."
            return await self.get_error_answer(question, refined_question, user, chat_id, is_new_chat, chat_memory, error_message)

        generated_charts = {}
        charts_data_obj = {}

        if rows and chart_types:
            charts_data_obj = await self.db_mini_prompt_service.prepare_chart_data(rows_summary, refined_question, chart_types)

            # return charts_data_obj
            for chart_type, chart_data in charts_data_obj.items():

                if chart_data:
                    chart_path = Helper.generate_chart(
                        chart_data=chart_data,
                        chart_type=chart_type,
                        output_dir="app/charts",
                        chart_name=chart_type
                    )
                    if chart_path and not chart_path.startswith("Could not"):
                        public_url = chart_path.replace("app/charts", f'{settings.BACKEND_BASE_URL}/charts').replace("\\", "/")
                        generated_charts[chart_type] = public_url


        # return final_sql_query, rows, generated_charts
        response = await self.db_prompt_service.format_response(
            question=refined_question,
            data_summary=rows_summary,
            charts=generated_charts,
            error_message=error_message,
            generate_title=is_new_chat
        )

        print(response)
    
        title = response.get('title', "")
        text = response.get('text', "")
        html = response.get('html', "")

        await self.chat_memory_service.add_assistant_message(chat_id, question)
        self.intranet_db_service.add_chat_message(chat_id, 'HUMAN', question)

        chat = None
        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, 'CTC', user.email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, html)


        # data_converted = Helper.convert_decimals_in_obj(rows)
        # data_str = json.dumps(data_converted)
                
        self.intranet_db_service.add_chat_message(chat_id, 'ASSISTANT', text, html, "", final_sql_query)

        return {
            "data": {
                "type": "DATABSE",
                "title": title,
                "answer": html,
                "chat_id": chat_id,
                "chat": chat,
                "charts": generated_charts,
                "sql_query": final_sql_query,
                "sql_results": rows,
            }
        }
    

    
    async def get_error_answer(self, question: str, refined_question: str, user: SendChatRequestUser, chat_id: str,chat, is_new_chat: bool, chat_memory: ConversationBufferMemory, error_message: str):
       
        error_response = await self.chat_prompt_service.generate_conversational_error(refined_question, error_message)
        text_answer = error_response.get('text_answer', '')
        html_answer = error_response.get('html_answer', '')
        title = error_response.get('title', [])

        await self.chat_memory_service.add_human_message(chat_id, question)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", question)

        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, "POLICIES_PROCEDURES", user.email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, text_answer)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", text_answer, html_answer)

        return {
            "data": {
                "type": "ERROR",
                "title": title,
                "answer": html_answer,
                "chat_id": chat_id,
                "chat": chat,
            }
        }
    
        