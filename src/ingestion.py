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
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def get_ingested_sources(self):
        if self.collection.count() == 0:
            return set()
        results = self.collection.get(include=["metadatas"])
        return {m.get("source") for m in results["metadatas"] if m.get("source")}

    def load(self, urls):
        docs = [WebBaseLoader(url).load() for url in urls]
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
        print(f"Stored {len(doc_splits)} chunks in Chroma")
        return vectorstore

    def get_retriever(self):
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.models.embed_model,
            client=self.client
        )
        return vectorstore.as_retriever(search_kwargs={"k": 2})

    def run(self, urls=None):
        urls = urls or URLS
        already_ingested = self.get_ingested_sources()
        new_urls = [url for url in urls if url not in already_ingested]

        if new_urls:
            print(f"Ingesting {len(new_urls)} new URLs...")
            docs_list = self.load(new_urls)
            doc_splits = self.chunk(docs_list)
            self.store(doc_splits)
        else:
            print("All URLs already ingested. Skipping.")

        return self.get_retriever()


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
