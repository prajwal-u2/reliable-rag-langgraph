from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import EMBED_MODEL, LLM_MODEL
from dotenv import load_dotenv

class Models:
    def __init__(self):
        load_dotenv()
        self.embed_model = FastEmbedEmbeddings(model_name=EMBED_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        self.json_llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")
        self.web_search_tool = TavilySearchResults(k=3)
