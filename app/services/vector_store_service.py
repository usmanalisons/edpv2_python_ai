from pymilvus import connections, Collection
from langchain_milvus import Milvus
from langchain_core.documents import Document
from app.core.config import settings
from langchain_openai.embeddings import OpenAIEmbeddings
from threading import Lock
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.connection_args = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
        }
        self.collection_names = settings.COLLECTION_NAMES
        self.embedding_function = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        self.vector_store = {}
        self._lock = Lock()

    def _init_connection(self):
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                host=self.connection_args["host"],
                port=str(self.connection_args["port"]),
                user=self.connection_args["user"],
                password=self.connection_args["password"]
            )
            logger.info("Connected to Milvus with alias 'default'.")

    def _init_vector_store(self, collection_key):
        self._init_connection()

        collection_name = self.collection_names.get(collection_key)
        if not collection_name:
            raise ValueError(f"Collection key '{collection_key}' is not configured in settings.")

        if collection_key not in self.vector_store:
            with self._lock:
                if collection_key not in self.vector_store:
                    vector_store_instance = Milvus(
                        embedding_function=self.embedding_function,
                        connection_args=self.connection_args,
                        collection_name=collection_name,
                        auto_id=True,
                    )
                    self.vector_store[collection_key] = vector_store_instance

                    try:
                        collection = Collection(collection_name)
                        collection.load()
                        logger.info(f"Collection '{collection_name}' preloaded into memory.")
                    except Exception as e:
                        logger.error(f"Error loading collection '{collection_name}': {str(e)}")

    def get_vector_store(self, collection_key):
        """
        Retrieves the vector store for the given collection key.
        """
        self._init_vector_store(collection_key)
        return self.vector_store[collection_key]

    def add_embeddings(self, collection_key, metadatas, chunks, batch_size=100):
        """
        Adds embeddings to Milvus in batches.
        """
        vector_store = self.get_vector_store(collection_key)

        documents = [
            Document(page_content=chunk, metadata={key: value or "" for key, value in (metadata or {}).items()})
            for chunk, metadata in zip(chunks, metadatas)
        ]
        vector_store.add_documents(documents)

    def search_embeddings(self, collection_key, query, top_k=20, filters=None, score_threshold=None):
        vector_store = self.get_vector_store(collection_key)

        search_params = {"metric_type": "L2", "params": {"ef": max(top_k, 50)}}

        try:
            search_results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                expr=filters,
                search_params=search_params,
            )

           

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def retrieve_as_retriever(self, collection_key, search_type="similarity_score_threshold", filters=None):
        vector_store = self.get_vector_store(collection_key)

        # Ensure `ef` is properly set to be greater than or equal to `k`
        k_value = 20
        ef_value = max(k_value, 50)  # ef should always be >= k

        search_kwargs = {
            "k": k_value,
            "score_threshold": 0.5,
            "search_params": {"metric_type": "COSINEM", "params": {"ef": ef_value}}
        }

        if filters:
            search_kwargs["filters"] = filters

        try:
            retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            return retriever
        except Exception as e:
            logger.error(f"Retriever initialization failed: {str(e)}")
            raise



    def release_collection(self, collection_key):
        self._init_connection()

        if collection_key in self.vector_store:
            collection_name = self.collection_names.get(collection_key)
            try:
                collection = Collection(collection_name)
                collection.release()
                logger.info(f"Collection '{collection_name}' released from memory.")
            except Exception as e:
                logger.error(f"Error releasing collection '{collection_name}': {str(e)}")


    def recreate_index(self, collection_key, index_params=None):
        self._init_connection()

        collection_name = self.collection_names.get(collection_key)
        if not collection_name:
            raise ValueError(f"Collection key '{collection_key}' is not configured in settings.")

        try:
            collection = Collection(collection_name)

            return collection

            if collection.indexes:
                collection.drop_index()
                logger.info(f"Index dropped for collection '{collection_name}'.")

            if not index_params:
                # index_params = {
                #     "index_type": "HNSW",
                #     "metric_type": "L2",
                #     "params": {"M": 16, "efConstruction": 50}
                # }

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 48, "efConstruction": 200},
                }

            collection.create_index(
                field_name="vector", 
                index_params=index_params,
            )
            logger.info(f"Index created for collection '{collection_name}' with params: {index_params}.")

            collection.load()
            logger.info(f"Collection '{collection_name}' loaded into memory.")

        except Exception as e:
            logger.error(f"Failed to recreate index for collection '{collection_name}': {str(e)}")
            raise

