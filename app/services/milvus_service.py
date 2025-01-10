from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from app.core.config import settings
import logging

class MilvusService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.collection_mapping = settings.COLLECTION_NAMES
        self.vector_dimension = settings.VECTOR_DIMENSION
        self._connect_to_milvus()
        self.collections = self.create_collections()
        self._initialized = True

    def _connect_to_milvus(self):
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
            )
            logging.info("Connected to Milvus.")
        else:
            logging.info("Already connected to Milvus.")

    def create_collections(self):
        collections = {}
        for collection_key, collection_name in self.collection_mapping.items():
            if collection_name in utility.list_collections():
                logging.info(f"Collection '{collection_name}' already exists.")
                collections[collection_key] = Collection(collection_name)
            else:
                logging.info(f"Collection '{collection_name}' does not exist. Creating it...")
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                ]
                schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name} embeddings")
                collection = Collection(name=collection_name, schema=schema)

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 48, "efConstruction": 200},
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logging.info(f"Collection '{collection_name}' created successfully.")
                collections[collection_key] = collection
        return collections

    def insert_embeddings(self, collection_key, embeddings, metadatas, texts):
       
        if collection_key not in self.collections:
            logging.error(f"Logical collection '{collection_key}' does not exist.")
            return False
        try:
            data = [
                embeddings,
                texts,
                metadatas,
            ]
            collection = self.collections[collection_key]
            collection.insert(data)
            if utility.load_state(self.collection_mapping[collection_key]) != "Loaded":
                collection.load()
            return True
        except Exception as e:
            logging.error(f"Error Inserting Data In Milvus: {e}")
            return False

    def search_embeddings(self, collection_key, embedding, top_k=50, expr=None):
        if collection_key not in self.collections:
            logging.error(f"Logical collection '{collection_key}' does not exist.")
            return []
        collection = self.collections[collection_key]
        if utility.load_state(self.collection_mapping[collection_key]) != "Loaded":
            collection.load()
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 512},
        }
        results = collection.search(
            data=[embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "document_title", "page_number"],
        )

        # Prepare results in a structured format
        output = []
        for hits in results:
            for hit in hits:
                # Extracting fields directly from the entity property
                result_data = {
                    "id": hit.id,  # Milvus ID of the hit
                    "distance": hit.distance,  # Similarity score
                    "text": hit.entity.get("text") if hasattr(hit.entity, "get") else hit.entity["text"],
                    "document_title": hit.entity.get("document_title") if hasattr(hit.entity, "get") else hit.entity["document_title"],
                    "page_number": hit.entity.get("page_number") if hasattr(hit.entity, "get") else hit.entity["page_number"],
                }
                output.append(result_data)

        return output
    def disconnect(self):
        for collection_key, collection_name in self.collection_mapping.items():
            if utility.load_state(collection_name) == "Loaded":
                self.collections[collection_key].release()
        if connections.has_connection("default"):
            connections.disconnect(alias="default")
            logging.info("Disconnected from Milvus.")
