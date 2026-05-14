import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.config import URLS, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from src.models import Models


class DataIngestion:
    def __init__(self):
        self.models = Models()
        self.client = chromadb.HttpClient(host="localhost", port=8000)

    def load(self):
        docs = [WebBaseLoader(url).load() for url in URLS]
        docs_list = [item for sublist in docs for item in sublist]
        print(f"Loaded {len(docs_list)} documents")
        return docs_list

    def chunk(self, docs_list):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        doc_splits = splitter.split_documents(docs_list)
        print(f"Created {len(doc_splits)} chunks")
        return doc_splits

    def store(self, doc_splits):
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=self.models.embed_model,
            collection_name=COLLECTION_NAME,
            client=self.client
        )
        print("Stored documents in Chroma DB")
        return vectorstore.as_retriever(search_kwargs={"k":2})

    def run(self):
        docs_list = self.load()
        doc_splits = self.chunk(docs_list)
        retriever = self.store(doc_splits)
        print("Ingestion complete.")
        return retriever


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
