from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class TextProcessorService:
    def __init__(self, model_name: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pages(self, pages, document_data):
        """
        Process a list of pages into manageable text chunks and corresponding metadata.
        """
        document_chunks = []
        document_metadatas = []

        for page in pages:
            page_number = page["page_number"]
            text = page["text"]

            chunks = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(chunks, start=1):
                document_chunks.append(chunk_text)
                metadata = {
                    "page_number": page_number,
                    "chunk_number": i,
                    **document_data
                }
                document_metadatas.append(metadata)

        return document_chunks, document_metadatas
