import os
import logging
from uuid import uuid4
from fastapi import UploadFile
from app.utils.file_loader import load_text_from_file
from app.services.text_processor_service import TextProcessorService
from app.services.document_search_prompt_service import DocumentSearchPromptService
from app.utils.helper import Helper
from app.core.config import settings
from app.core.constants import MAX_TOKENS_PER_REQUEST
from app.services.sql_db_service import SQLDatabaseService
from app.services.chat_memory_service import ChatMemoryService
from app.services.chroma_db_service import ChromaDBService

class DocumentSearchLogic:
    def __init__(self, chroma_db_service, chat_memory_service):
        self.chroma_db_service: ChromaDBService = chroma_db_service
        self.chat_memory_service: ChatMemoryService = chat_memory_service
        self.intranet_db_service = SQLDatabaseService(connection_name="intranet")
        self.quantum_db_service = SQLDatabaseService(connection_name="quantum")
        self.text_processor_service = TextProcessorService(model_name=settings.OPENAI_EMBEDDING_MODEL)
        self.prompt_service = DocumentSearchPromptService(model_name="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

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
            self.intranet_db_service.update_policies_procedures_last_embedding_at(processed_document_ids)

        return {
            "data": {
                "processed_documents_count": len(processed_document_ids),
                "processed_document_ids": processed_document_ids,
                "missing_document_count": len(missing_document_ids),
                "missing_document_ids": missing_document_ids,
            }
        }

    async def process_oracle_trainings_from_db(self):
        collection_key = "oracleTrainings"
        await self.chroma_db_service.remove_collection(collection_key)
        trainings = self.quantum_db_service.get_oracle_trainings_from_db()

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

        human_messages = await self.chat_memory_service.get_human_messages(chat_id)
        
        refined_query = await self.prompt_service.refine_user_question(query, human_messages)

        filters = Helper.filter_policies_procedures(email, department_code, company_code)

        retriever = await self.chroma_db_service.retrieve_as_retriever(
            collection_key=collection_key,
            filter_dict=filters
        )

        relevant_docs = retriever.invoke(
            input=refined_query,
        )

        contexts = [doc.page_content for doc in relevant_docs]
        combined_context = " ".join(contexts)

        # return refined_query, relevant_docs

        answer_text, answer_from_context = await self.prompt_service.generate_text_answer(query, combined_context, chat_memory, "Policies & Procedures")

        # return answer_text, relevant_docs, answer_from_context

        answer_html, title = await self.prompt_service.format_text_answer(query, refined_query, generate_title=is_new_chat)

        await self.chat_memory_service.add_human_message(chat_id, query)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", query)

        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, "POLICIES_PROCEDURES", email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, answer_text)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", answer_text, answer_html)

        return {
            "data": {
                "type": "text",
                "answer": answer_html,
                "show_reference": answer_from_context,
                "title": title,
                "chat_id": chat_id,
                "relevant_docs": relevant_docs,
                "combined_context": combined_context,
                "chat": chat
            }
        }


    async def handle_oracle_trainings_search(self, email, query, chat_id=None):        
        collection_key = "oracleTrainings"
        chat_memory = None
        is_new_chat = False
        chat = None

        # memory_summary = await self.chat_memory_service.get_chat_summary(chat_id)
        # all_memories = await self.chat_memory_service.get_all_memories()

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

        human_messages = await self.chat_memory_service.get_human_messages(chat_id)

        refined_query = await self.prompt_service.refine_user_question(query, human_messages)

        retriever = await self.chroma_db_service.retrieve_as_retriever(
            collection_key=collection_key,
        )

        relevant_docs = retriever.invoke(
            input=refined_query,
        )

        contexts = [doc.page_content for doc in relevant_docs]
        combined_context = " ".join(contexts)

        answer_text, answer_from_context = await self.prompt_service.generate_text_answer(refined_query, combined_context, chat_memory, 'Oracle Trainings Wave 2')
       
    
        answer_html, title = await self.prompt_service.format_text_answer(answer_text, refined_query, generate_title=is_new_chat)

        await self.chat_memory_service.add_human_message(chat_id, query)
        self.intranet_db_service.add_chat_message(chat_id, "HUMAN", query)

        if is_new_chat:
            chat = self.intranet_db_service.add_chat(chat_id, "ORACLE_TRAININGS", email, title)

        await self.chat_memory_service.add_assistant_message(chat_id, answer_text)
        self.intranet_db_service.add_chat_message(chat_id, "ASSISTANT", answer_text, answer_html)

        return {
            "data": {
                "type": "text",
                "answer": answer_html,
                "show_reference": answer_from_context,
                "title": title,
                "chat_id": chat_id,
                "relevant_docs": relevant_docs,
                "combined_context": combined_context,
                "chat": chat,
            }
        }
