import asyncio
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class ChromaDBService:
    def __init__(self, persist_directory: str, collection_names: dict, model_name: str):
        self.persist_directory = persist_directory
        self.collection_names = collection_names
        self.embedding_model = OpenAIEmbeddings(model=model_name)

    async def get_all_collections(self):
        collections = {}
        for collection_key in self.collection_names:
            try:
                collection = await self.get_collection(collection_key)
                collections[collection_key] = collection
            except KeyError:
                print(f"Collection '{collection_key}' not found.")
        return collections

    async def get_collection(self, collection_key: str):
        if collection_key not in self.collection_names:
            raise KeyError(f"Key '{collection_key}' not found in COLLECTION_NAMES.")
        collection_name = self.collection_names[collection_key]
        return await run_in_threadpool(
            Chroma,
            collection_name=collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
        )

    async def add_embeddings(self, collection_key, metadatas, chunks, batch_size=100):
        db = await self.get_collection(collection_key)
        documents = [
            Document(page_content=chunk, metadata={key: value or "" for key, value in (metadata or {}).items()})
            for chunk, metadata in zip(chunks, metadatas)
        ]
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await run_in_threadpool(db.add_documents, batch)
        print(f"Successfully added {len(documents)} documents to the '{self.collection_names[collection_key]}' collection.")

    async def search_embedding(self, query: str, collection_key: str = "default", k: int = 5, ef_value: int = 128):
        db = await self.get_collection(collection_key)
        search_kwargs = {
            "k": k,
            "search_params": {
                "metric_type": "COSINE",
                "params": {"ef": ef_value}
            }
        }
        return await run_in_threadpool(db.similarity_search_with_score, query, **search_kwargs)

    async def retrieve_as_retriever(self, collection_key: str = "default"):
        db = await self.get_collection(collection_key)
        return await run_in_threadpool(
            db.as_retriever,
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.5},
        )

    async def release_collection(self, collection_key: str):
        db = await self.get_collection(collection_key)
        await run_in_threadpool(db.unload)
        print(f"Successfully released collection '{self.collection_names[collection_key]}'.")

    async def remove_collection(self, collection_key: str):
        db = await self.get_collection(collection_key)
        await run_in_threadpool(db.delete_collection)
        print(f"Successfully removed collection '{self.collection_names[collection_key]}'.")
