from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models import Models


class RetrievalGrader:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of a retrieved document to a user question.
                            If the document contains keywords related to the user question, grade it as relevant. 
                            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                            Provide the binary score as a JSON with a single key 'score' and no premable or explaination."""),
            ("human", "Here is the retrieved document: {document}\nHere is the user question: {question}"),
        ])
        self.models = Models()
    
    def run(self, document, question):
        grader = self.prompt | self.models.json_llm | JsonOutputParser()
        return grader.invoke({"document": document, "question": question})
    

class HallucinationGrader:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
                        Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
                        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. """),
            ("human", """Here are the facts: {document}\n
                        Here is the answer: {generation}"""),
        ])
        self.models = Models()
    
    def run(self, document, generation):
        grader = self.prompt | self.models.json_llm | JsonOutputParser()
        return grader.invoke({"document": document, "generation": generation})


class AnswerGrader:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is useful to resolve a question.
                        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
                        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", """Here is the question: {question}\n
                        Here is the answer: {generation}"""),
        ])
        self.models = Models()
    
    def run(self, question, generation):
        grader = self.prompt | self.models.json_llm | JsonOutputParser()
        return grader.invoke({"document": question, "answer": generation})