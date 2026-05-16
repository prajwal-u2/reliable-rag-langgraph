from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from src.models import Models
from src.ingestion import DataIngestion
from src.graders import AnswerGrader, RetrievalGrader, HallucinationGrader
from src.router import Router
from src.generator import AnswerGenerator
from langchain_community.tools.tavily_search import TavilySearchResults


class GraphState(TypedDict):
    question: str
    documents: List[str]
    web_search: str
    generation: str


class Node():
    def __init__(self):
        self.models = Models()
        self.ingestion = DataIngestion()
        self.retriever = self.ingestion.get_retriever()
        self.generator = AnswerGenerator()
        self.retrieval_grader = RetrievalGrader()
        self.web_search_tool = TavilySearchResults(k=3)


    def retrieve(self, state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVER---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = self.generator.run(question, documents)
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
        print("---GRADE DOCUMENTS---")

        question = state["question"]
        documents = state["documents"]
        web_search = 'No'
        filtered_docs = []
        for doc in documents:
            grade = self.retrieval_grader.run(doc.page_content, question)
            if grade["score"].lower() == "yes":
                filtered_docs.append(doc)
                print("---Relevant---")
            else:
                web_search = 'Yes'
                print("-Not Relevant-")
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB Search---")
        question = state["question"]
        documents = state["documents"]

        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}
    

class Edge:
    def __init__(self):
        self.router = Router()
        self.hallucination_grader = HallucinationGrader()
        self.answer_grader = AnswerGrader()

    def route_question(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.router.run(question)
        print(source["datasource"])

        if source["datasource"].lower() == "web_search":
            return "websearch"
        return "vectorstore"
    
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        if web_search == "Yes":
            return "websearch"
        return "generate"
    
    def verify_generations(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        score = self.hallucination_grader.run(documents, generation)
        grade = score['score']

        if grade == 'yes':
            print("--No Hallucination--")
            score = self.answer_grader.run(question, generation)
            grade = score['score']
            if grade == "yes":
                print("-Relevant Answer-")
                return "relevant"
            else:
                return "irrelevant"
        else:
            print("--Hallucination--")
            return "not supported"
        

    
    
        