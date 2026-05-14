import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.models import embed_model
from src.ingestion import doc_splits
from dotenv import load_dotenv

load_dotenv()

# connect
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

existing = [i.name for i in pc.list_indexes()]
if "rag-index" not in existing:
    pc.create_index(
        name="rag-index",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# insert chunks
PineconeVectorStore.from_documents(
    documents=doc_splits,
    embedding=embed_model,
    index_name="rag-index",
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)

print("Done!")
