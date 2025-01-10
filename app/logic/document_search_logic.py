# import os
# import logging
# from fastapi import UploadFile
# from app.utils.file_loader import load_text_from_file
# from app.services.sql_db_service import SQLDatabaseService
# from app.services.text_processor_service import TextProcessorService
# from app.services.vector_store_service import VectorStoreService
# from app.services.query_service import QueryService
# from app.services.openai_service import OpenAIService
# from app.utils.helper import Helper
# from app.core.config import settings
# from app.core.constants import MAX_TOKENS_PER_REQUEST
# from app.dependencies import get_chromedb_service

# from app.services.chat_memory_manager_service import ChatMemoryManager
# from uuid import uuid4


# class DocumentSearchLogic:
#     def __init__(self):
#         self.text_processor_service = TextProcessorService(model_name=settings.OPENAI_EMBEDDING_MODEL)
#         self.vector_store_service = VectorStoreService(
#             host=settings.MILVUS_HOST,
#             port=settings.MILVUS_PORT,
#             user=settings.MILVUS_USER,
#             password=settings.MILVUS_PASSWORD,
#         )
#         self.query_service = QueryService(model_name="gpt-4o", api_key=settings.OPENAI_API_KEY)
#         self.openAIService = OpenAIService()

#         self.chroma_db_service = get_chromedb_service()


#     def get_employee_chats(self, email, type):
#         sql_service = SQLDatabaseService(connection_name="intranet")
#         return sql_service.get_employee_chats(email, type)
    
#     def get_chat_messages(self, chat_id):
#         sql_service = SQLDatabaseService(connection_name="intranet")
#         return sql_service.get_chat_messages(chat_id)

#     def get_policies_procedures_from_db(self, file_type):
#         sql_service = SQLDatabaseService(connection_name="intranet")
#         return sql_service.get_policies_procedures_from_db(file_type)

#     def get_oracle_trainings_from_db(self):
#         sql_service = SQLDatabaseService(connection_name="quantum")
#         return sql_service.get_oracle_trainings_from_db()

#     def update_policies_procedures_last_embedding_at(self, document_ids):
#         sql_service = SQLDatabaseService(connection_name="intranet")
#         return sql_service.update_policies_procedures_last_embedding_at(document_ids)

#     async def handle_file_upload(self, file: UploadFile):

#         file_location = f"app/data/{file.filename}"
#         os.makedirs(os.path.dirname(file_location), exist_ok=True)
#         with open(file_location, "wb+") as f:
#             f.write(await file.read())

#         pages = load_text_from_file(file_location)

#         return pages

#         text = ""
#         for page in pages:
#             text = page["text"]
#         return text
#         if not pages:
#             return {"message": "Could not extract text or invalid file type."}

#         record = {
#             "document_id": 0,
#             "document_type": "Policies & Procedure",
#             "document_title": file.filename,
#             "document_path": file_location,
#             "access_company": "All",
#             "access_department": "All",
#             "access_employee_emails": "",
#         }

#         chunks, metadatas = self.text_processor_service.process_pages(pages, record)
#         collection_key = "policiesProcedures"
#         self.vector_store_service.add_embeddings(
#                 collection_key="policiesProcedures",
#                 metadatas=metadatas,
#                 chunks=chunks,
#             )

#         return {
#             "message": "File processed and embeddings stored successfully.",
#             "chunks_processed": len(chunks),
#         }

#     async def process_policies_procedures_from_db(self):
#         collection_key="policiesProcedures"
#         await self.chroma_db_service.remove_collection(collection_key)
#         documents = self.get_policies_procedures_from_db("pdf")

#         if not documents:
#             return {"data": {"message": "No document records found"}}

#         missing_document_ids = []
#         processed_document_ids = []
#         all_chunks, all_metadatas = [], []

#         for record in documents:
#             document_id = record["document_id"]
#             file_path = f"app/{record['document_path']}"
            
#             if not os.path.exists(file_path):
#                 missing_document_ids.append(document_id)
#                 continue

#             pages = load_text_from_file(file_path)
#             if not pages:
#                 missing_document_ids.append(document_id)
#                 continue

#             processed_document_ids.append(document_id)
#             chunks, metadatas = self.text_processor_service.process_pages(pages, record)
#             all_chunks.extend(chunks)
#             all_metadatas.extend(Helper.prepare_policies_procedures_metadatas(chunks, metadatas))

#         batches = Helper.batch_process(all_chunks, all_metadatas, max_tokens=MAX_TOKENS_PER_REQUEST)
#         for batch_chunks, batch_metadatas in batches:

#             await self.chroma_db_service.add_embeddings(
#                 collection_key=collection_key,
#                 metadatas=batch_metadatas,
#                 chunks=batch_chunks,
#             )

#         if processed_document_ids:
#             self.update_policies_procedures_last_embedding_at(processed_document_ids)

#         return {
#             "data": {
#                 "processed_documents_count": len(processed_document_ids),
#                 "processed_document_ids": processed_document_ids,
#                 "missing_document_count": len(missing_document_ids),
#                 "missing_document_ids": missing_document_ids,
#             }
#         }

#     async def process_oracle_trainings_from_db(self):
#         collection_key = "oracleTrainings"
#         self.chroma_db_service.remove_collection(collection_key)
#         trainings = self.get_oracle_trainings_from_db()

#         if not trainings:
#             return {"data": {"message": "No training records found"}}

#         missing_document_ids = []
#         processed_document_ids = []
#         all_chunks, all_metadatas = [], []

#         for record in trainings:
#             document_id = record["id"]
#             file_path = f"app/{record['url']}"

#             if not os.path.exists(file_path):
#                 missing_document_ids.append(document_id)
#                 continue

#             pages = load_text_from_file(file_path)
#             if not pages:
#                 missing_document_ids.append(document_id)
#                 continue

#             processed_document_ids.append(document_id)
#             chunks, metadatas = self.text_processor_service.process_pages(pages, record)
#             all_chunks.extend(chunks)
#             all_metadatas.extend(Helper.prepare_oracle_trainings_metadatas(chunks, metadatas))

#         batches = Helper.batch_process(all_chunks, all_metadatas, max_tokens=MAX_TOKENS_PER_REQUEST)

#         for batch_chunks, batch_metadatas in batches:

#             self.chroma_db_service.add_embeddings(
#                 collection_key=collection_key,
#                 metadatas=batch_metadatas,
#                 chunks=batch_chunks,
#             )
#         return {
#             "data": {
#                 "processed_documents_count": len(processed_document_ids),
#                 "processed_document_ids": processed_document_ids,
#                 "missing_document_count": len(missing_document_ids),
#                 "missing_document_ids": missing_document_ids,
#             }
#         }

#     def filter_policies_procedures(self, email: str, department_code: str, company_code: str,):
#         conditions = [
#             '(access_company == "All" and access_department == "All")',
#             f'(access_employee_emails LIKE "%{email}%")',
#             f'(access_company == "All" and access_department LIKE "%{department_code}%")',
#             f'(access_company LIKE "%{company_code}%" and access_department == "All")',
#             f'(access_company LIKE "%{company_code}%" and access_department LIKE "%{department_code}%")',
#         ]
#         return " or ".join(conditions)
    
#     async def handle_policies_procedure_search(self, email: str, department_code: str, company_code: str, query: str, chat_id: str = None):
#         collection_key = "policiesProcedures"
#         chat = None
#         previous_message = []
#         is_new_chat = False
#         chat_memory = None

#         intranet_db_service = SQLDatabaseService(connection_name="intranet")
#         memory_manager = ChatMemoryManager(openai_api_key=settings.OPENAI_API_KEY)

#         if not chat_id:
#             is_new_chat = True
#             chat_id = str(uuid4())
#             chat_memory = memory_manager.create_memory(chat_id, [])
#         else:
#             chat = intranet_db_service.get_chat_by_id(chat_id)
#             if not chat:
#                 return None
#             if memory_manager.has_memory(chat_id):
#                 chat_memory = memory_manager.get_memory(chat_id)
#             else:
#                 previous_message = intranet_db_service.get_chat_messages(chat_id)
#                 chat_memory = memory_manager.create_memory(chat_id, previous_message)

#         chatMessages = chat_memory.chat_memory.messages

#         human_messages = [msg.content for msg in chatMessages if type(msg).__name__ == 'HumanMessage']
#         human_messages_string = " ".join(human_messages) + " " + query

       
#         retriever = await self.chroma_db_service.retrieve_as_retriever(
#             collection_key=collection_key,
#         )
#         relevant_docs = retriever.invoke(human_messages_string)

#         contexts = [doc.page_content for doc in relevant_docs]
        
#         combined_context = " ".join(contexts)

#         answer, title, complete_response = self.query_service.generate_contextual_answer(query, combined_context, generate_title=is_new_chat, memory=chat_memory)

#         memory_manager.add_user_message(chat_id, query)
#         intranet_db_service.add_chat_message(chat_id, 'HUMAN', query)

#         if is_new_chat:
#             chat = intranet_db_service.add_chat(chat_id, 'POLICIES_PROCEDURES', email, title)

#         memory_manager.add_assistant_message(chat_id, answer)
#         intranet_db_service.add_chat_message(chat_id, 'ASSISTANT', answer)

#         return {
#             "data": {
#                 "type": "text",
#                 "answer": answer,
#                 "title": title,
#                 "chat": chat,
#                 "relevant_docs": relevant_docs,
#                 "chatMessages": chatMessages,
#                 "combined_context": combined_context,
#                 "complete_response": complete_response
#             }
#         }

    
#     async def handle_oracle_trainings_search(self, email: str, query: str, chat_id: str = None):
#         collection_key = "oracleTrainings"
#         chat = None
#         previous_message = []
#         is_new_chat = False
#         chat_memory = None

#         intranet_db_service = SQLDatabaseService(connection_name="intranet")
#         memory_manager = ChatMemoryManager(openai_api_key=settings.OPENAI_API_KEY)

#         if not chat_id:
#             is_new_chat = True
#             chat_id = str(uuid4())
#             chat_memory = memory_manager.create_memory(chat_id, [])
#         else:
#             chat = intranet_db_service.get_chat_by_id(chat_id)
#             if not chat:
#                 return None
#             if memory_manager.has_memory(chat_id):
#                 chat_memory = memory_manager.get_memory(chat_id)
#             else:
#                 previous_message = intranet_db_service.get_chat_messages(chat_id)
#                 chat_memory = memory_manager.create_memory(chat_id, previous_message)

#         chatMessages = chat_memory.chat_memory.messages

#         human_messages = [msg.content for msg in chatMessages if type(msg).__name__ == 'HumanMessage']
#         human_messages_string = " ".join(human_messages) + " " + query
#         retriever = self.chroma_db_service.retrieve_as_retriever(
#             collection_key=collection_key,
#         )

#         relevant_docs = retriever.invoke(human_messages_string)

#         contexts = [doc.page_content for doc in relevant_docs]

#         combined_context = " ".join(contexts)

#         answer, title, complete_response = self.query_service.generate_contextual_answer(query, combined_context, generate_title=is_new_chat,  memory=chat_memory)

#         memory_manager.add_user_message(chat_id, query)
#         intranet_db_service.add_chat_message(chat_id, 'HUMAN', query)

#         if is_new_chat:
#             chat = intranet_db_service.add_chat(chat_id, 'ORACLE_TRAININGS', email, title)

#         memory_manager.add_assistant_message(chat_id, answer)
#         intranet_db_service.add_chat_message(chat_id, 'ASSISTANT', answer)

#         return {
#             "data": {
#                 "type": "text",
#                 "answer": answer,
#                 "title": title,
#                 "chat": chat,
#                 "relevant_docs": relevant_docs,
#                 "chatMessages": chatMessages,
#                 "combined_context": combined_context,
#                 "complete_response": complete_response
#             }
#         }



import os
import logging
from uuid import uuid4
from fastapi import UploadFile
from app.utils.file_loader import load_text_from_file
from app.services.text_processor_service import TextProcessorService
from app.services.query_service import QueryService
from app.utils.helper import Helper
from app.core.config import settings
from app.core.constants import MAX_TOKENS_PER_REQUEST

class DocumentSearchLogic:
    def __init__(self, chroma_db_service, memory_manager, intranet_db_service, quantum_db_service):
        self.chroma_db_service = chroma_db_service
        self.memory_manager = memory_manager
        self.intranet_db_service = intranet_db_service
        self.text_processor_service = TextProcessorService(model_name=settings.OPENAI_EMBEDDING_MODEL)
        self.query_service = QueryService(model_name="gpt-4o", api_key=settings.OPENAI_API_KEY)

    def get_employee_chats(self, email, type):
        return self.intranet_db_service.get_employee_chats(email, type)
    
    def get_chat_messages(self, chat_id):
        return self.intranet_db_service.get_chat_messages(chat_id)

    def get_policies_procedures_from_db(self, file_type):
        return self.intranet_db_service.get_policies_procedures_from_db(file_type)

    def get_oracle_trainings_from_db(self):
        return self.intranet_db_service.get_oracle_trainings_from_db()

    def update_policies_procedures_last_embedding_at(self, document_ids):
        return self.intranet_db_service.update_policies_procedures_last_embedding_at(document_ids)

    async def handle_file_upload(self, file: UploadFile):
        file_location = f"app/data/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb") as f:
            f.write(await file.read())

        pages = load_text_from_file(file_location)

        if not pages:
            return {"message": "Could not extract text or invalid file type."}

        return pages

    async def process_policies_procedures_from_db(self):
        collection_key = "policiesProcedures"
        await self.chroma_db_service.remove_collection(collection_key)
        documents = self.intranet_db_service.get_policies_procedures_from_db("pdf")

        if not documents:
            return {"data": {"message": "No document records found"}}

        missing_document_ids = []
        processed_document_ids = []
        all_chunks, all_metadatas = [], []

        for record in documents:
            document_id = record["document_id"]
            file_path = f"app/{record['document_path']}"

            if not os.path.exists(file_path):
                missing_document_ids.append(document_id)
                continue

            pages = load_text_from_file(file_path)
            if not pages:
                missing_document_ids.append(document_id)
                continue

            processed_document_ids.append(document_id)
            chunks, metadatas = self.text_processor_service.process_pages(pages, record)
            all_chunks.extend(chunks)
            all_metadatas.extend(Helper.prepare_policies_procedures_metadatas(chunks, metadatas))

        batches = Helper.batch_process(all_chunks, all_metadatas, max_tokens=MAX_TOKENS_PER_REQUEST)
        for batch_chunks, batch_metadatas in batches:
            await self.chroma_db_service.add_embeddings(
                collection_key=collection_key,
                metadatas=batch_metadatas,
                chunks=batch_chunks,
            )

        if processed_document_ids:
            self.db_service.update_policies_procedures_last_embedding_at(processed_document_ids)

        return {
            "data": {
                "processed_documents_count": len(processed_document_ids),
                "processed_document_ids": processed_document_ids,
                "missing_document_count": len(missing_document_ids),
                "missing_document_ids": missing_document_ids,
            }
        }

    async def process_oracle_trainings_from_db(self):
        """
        Processes Oracle trainings from the database.
        """
        collection_key = "oracleTrainings"
        await self.chroma_db_service.remove_collection(collection_key)
        trainings = self.intranet_db_service.get_oracle_trainings_from_db()

        if not trainings:
            return {"data": {"message": "No training records found"}}

        missing_document_ids = []
        processed_document_ids = []
        all_chunks, all_metadatas = [], []

        for record in trainings:
            document_id = record["id"]
            file_path = f"app/{record['url']}"

            if not os.path.exists(file_path):
                missing_document_ids.append(document_id)
                continue

            pages = load_text_from_file(file_path)
            if not pages:
                missing_document_ids.append(document_id)
                continue

            processed_document_ids.append(document_id)
            chunks, metadatas = self.text_processor_service.process_pages(pages, record)
            all_chunks.extend(chunks)
            all_metadatas.extend(Helper.prepare_oracle_trainings_metadatas(chunks, metadatas))

        batches = Helper.batch_process(all_chunks, all_metadatas, max_tokens=MAX_TOKENS_PER_REQUEST)

        for batch_chunks, batch_metadatas in batches:
            await self.chroma_db_service.add_embeddings(
                collection_key=collection_key,
                metadatas=batch_metadatas,
                chunks=batch_chunks,
            )

        return {
            "data": {
                "processed_documents_count": len(processed_document_ids),
                "processed_document_ids": processed_document_ids,
                "missing_document_count": len(missing_document_ids),
                "missing_document_ids": missing_document_ids,
            }
        }

    async def handle_policies_procedure_search(self, email: str, department_code: str, company_code: str, query: str, chat_id: str = None):
        collection_key = "policiesProcedures"
        chat_memory = None
        is_new_chat = False

        if not chat_id:
            is_new_chat = True
            chat_id = str(uuid4())
            chat_memory = await self.memory_manager.create_memory(chat_id, [])
        else:
            chat = self.intranet_db_service.get_chat_by_id(chat_id)
            if not chat:
                return {"message": "Chat ID not found."}

            if await self.memory_manager.has_memory(chat_id):
                chat_memory = await self.memory_manager.get_memory(chat_id)
            else:
                previous_messages = self.intranet_db_service.get_chat_messages(chat_id)
                chat_memory = await self.memory_manager.create_memory(chat_id, previous_messages)

        # Ensure `chat_memory` has the expected structure
        human_messages = [msg.content for msg in chat_memory.chat_memory.messages]
        human_messages_string = " ".join(human_messages) + " " + query

        retriever = await self.chroma_db_service.retrieve_as_retriever(
            collection_key=collection_key
        )
        relevant_docs = retriever.invoke(human_messages_string)

        contexts = [doc.page_content for doc in relevant_docs]
        combined_context = " ".join(contexts)

        answer, title, complete_response = self.query_service.generate_contextual_answer(
            query, combined_context, generate_title=is_new_chat, memory=chat_memory
        )

        await self.memory_manager.add_user_message(chat_id, query)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", query)

        if is_new_chat:
            self.intranet_db_service.add_chat(chat_id, "POLICIES_PROCEDURES", email, title)

        await self.memory_manager.add_assistant_message(chat_id, answer)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", answer)

        return {
            "data": {
                "type": "text",
                "answer": answer,
                "title": title,
                "chat_id": chat_id,
                "relevant_docs": relevant_docs,
                "combined_context": combined_context,
                "complete_response": complete_response,
            }
        }


    async def handle_oracle_trainings_search(self, email, query, chat_id=None):        
        collection_key = "oracleTrainings"
        chat_memory = None
        is_new_chat = False

        if not chat_id:
            is_new_chat = True
            chat_id = str(uuid4())
            chat_memory = self.memory_manager.create_memory(chat_id, [])
        else:
            chat = self.intranet_db_service.get_chat_by_id(chat_id)
            if not chat:
                return None
            chat_memory = self.memory_manager.get_memory(chat_id)

        human_messages = [msg.content for msg in chat_memory.chat_memory.messages]
        human_messages_string = " ".join(human_messages) + " " + query

        retriever = await self.chroma_db_service.retrieve_as_retriever(collection_key=collection_key)
        relevant_docs = retriever.invoke(human_messages_string)

        contexts = [doc.page_content for doc in relevant_docs]
        combined_context = " ".join(contexts)

        answer, title, complete_response = self.query_service.generate_contextual_answer(
            query, combined_context, generate_title=is_new_chat, memory=chat_memory
        )

        self.memory_manager.add_user_message(chat_id, query)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", query)

        if is_new_chat:
            self.intranet_db_service.add_chat(chat_id, "ORACLE_TRAININGS", email, title)

        self.memory_manager.add_assistant_message(chat_id, answer)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", answer)

        return {
            "data": {
                "type": "text",
                "answer": answer,
                "title": title,
                "chat_id": chat_id,
                "relevant_docs": relevant_docs,
                "combined_context": combined_context,
                "complete_response": complete_response,
            }
        }
