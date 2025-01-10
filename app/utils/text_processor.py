# # app/utils/text_processor.py

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from app.core.config import settings
# import tiktoken

# class TextProcessor:
#     """
#     Processes document pages into chunks for embedding, now using LangChain's built-in text splitter.
#     """
#     def __init__(self, model_name=settings.OPENAI_EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=200):
#         self.tokenizer = tiktoken.encoding_for_model(model_name)
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", " ", ""]
#         )

#     def process_pages(self, pages, document_data):
#         """
#         Given a list of pages with text content, this method splits each page's text 
#         into manageable chunks using the LangChain text splitter, and returns both 
#         chunks and their corresponding metadata.
#         """
#         document_chunks = []
#         document_metadatas = []

#         for page in pages:
#             page_number = page["page_number"]
#             text = page["text"]

#             # Use LangChain's text splitter to get chunks
#             chunks = self.splitter.split_text(text)
#             for i, chunk_text in enumerate(chunks, start=1):
#                 document_chunks.append(chunk_text)
#                 metadata = {
#                     "page_number": page_number,
#                     "chunk_number": i,
#                     **document_data
#                 }
#                 document_metadatas.append(metadata)

#         return document_chunks, document_metadatas
