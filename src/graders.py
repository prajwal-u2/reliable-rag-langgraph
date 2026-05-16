from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models import Models
from abc import ABC, abstractmethod


class BaseGrader(ABC):
    def __init__(self):    
        self.models = Models()
        self.prompt = self._build_prompt()
    
    @abstractmethod
    def _build_prompt(self) -> ChatPromptTemplate:
        pass

    @abstractmethod
    def run(self, **kwargs) -> dict:
        pass


class RetrievalGrader(BaseGrader):
    def _build_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of a retrieved document to a user question.
                            If the document contains keywords related to the user question, grade it as relevant. 
                            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                            Provide the binary score as a JSON with a single key 'score' and no premable or explaination."""),
            ("human", "Here is the retrieved document: {document}\nHere is the user question: {question}"),
        ])
    
    def run(self, document, question):
        grader = self.prompt | self.models.json_llm | JsonOutputParser()
        return grader.invoke({"document": document, "question": question})
    

class HallucinationGrader(BaseGrader):
    def _build_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
                        Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
                        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. """),
            ("human", """Here are the facts: {document}\n
                        Here is the answer: {generation}"""),
        ])
    
    def run(self, document, generation):
        grader = self.prompt | self.models.json_llm | JsonOutputParser()
        return grader.invoke({"document": document, "generation": generation})


class AnswerGrader(BaseGrader):
    def _build_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is useful to resolve a question.
                        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
                        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", """Here is the question: {question}\n
                        Here is the answer: {generation}"""),
        ])
    
    def run(self, question, generation):
        grader = self.prompt | self.models.json_llm | JsonOutputParser()
        return grader.invoke({"question": question, "generation": generation})