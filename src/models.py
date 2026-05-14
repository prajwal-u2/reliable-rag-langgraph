from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.config import EMBED_MODEL, LLM_MODEL


class Models:
    def __init__(self):
        self.embed_model = FastEmbedEmbeddings(model_name=EMBED_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        self.json_llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")
