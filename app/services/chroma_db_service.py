import asyncio
from typing import Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class ChromaDBService:
    def __init__(self, persist_directory: str, collection_names: Dict[str, str], model_name: str):
        self.persist_directory: str = persist_directory
        self.collection_names: Dict[str, str] = collection_names
        self.embedding_model = OpenAIEmbeddings(model=model_name)
        self._collection_cache: Dict[str, Chroma] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_all_collections(self) -> Dict[str, Chroma]:
        async with self._lock:
            collections: Dict[str, Chroma] = {}
            for collection_key in self.collection_names:
                try:
                    collections[collection_key] = await self.get_collection(collection_key)
                except KeyError:
                    print(f"Collection '{collection_key}' not found.")
            return collections

    async def get_collection(self, collection_key: str) -> Chroma:
        if collection_key not in self.collection_names:
            raise KeyError(f"Key '{collection_key}' not found in COLLECTION_NAMES.")
        async with self._lock:
            if collection_key not in self._collection_cache:
                collection_name: str = self.collection_names[collection_key]
                self._collection_cache[collection_key] = Chroma(
                    collection_name=collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                )
            return self._collection_cache[collection_key]

    async def add_embeddings(self, collection_key: str, metadatas: List[Dict[str, Optional[str]]], chunks: List[str]) -> None:
        db: Chroma = await self.get_collection(collection_key)
        documents: List[Document] = [
            Document(page_content=chunk, metadata={key: value or "" for key, value in (metadata or {}).items()})
            for chunk, metadata in zip(chunks, metadatas)
        ]
        await db.add_documents(documents)
        print(f"Successfully added {len(documents)} documents to the '{self.collection_names[collection_key]}' collection.")

    async def search_embedding(self, query: str, collection_key: str = "default", k: int = 5, ef_value: int = 128) -> List[tuple]:
        db: Chroma = await self.get_collection(collection_key)
        return await db.similarity_search_with_score(
            query, k=k, search_params={"metric_type": "COSINE", "params": {"ef": ef_value}}
        )

    async def retrieve_as_retriever(self, collection_key: str = "default", filter_dict: Optional[Dict[str, Dict[str, str]]] = None) -> Chroma:
        db: Chroma = await self.get_collection(collection_key)
        search_kwargs: Dict[str, object] = {"k": 10}
        # if filter_dict:
        #     search_kwargs["filter"] = filter_dict
        return db.as_retriever(search_kwargs=search_kwargs)

    async def release_collection(self, collection_key: str) -> None:
        async with self._lock:
            if collection_key in self._collection_cache:
                db: Chroma = self._collection_cache.pop(collection_key)
                db.unload()
                print(f"Successfully released collection '{self.collection_names[collection_key]}'.")

    async def remove_collection(self, collection_key: str) -> None:
        async with self._lock:
            if collection_key in self._collection_cache:
                db: Chroma = self._collection_cache.pop(collection_key)
                db.delete_collection()
                print(f"Successfully removed collection '{self.collection_names[collection_key]}'.")
