from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models import Models


class AnswerGenerator:

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks.
                        Use the following retrieved context to answer the question.
                        If you don't know the answer, just say that you don't know.
                        Use three sentences maximum and keep the answer concise."""),
            ("human", "Question: {question}\nContext: {context}"),
        ])
        self.models = Models()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, question, docs):
        context = self.format_docs(docs)
        rag_chain = self.prompt | self.models.llm | StrOutputParser()
        return rag_chain.invoke({"question": question, "context": context})

# Testing only
# if __name__ == "__main__":
#     from src.ingestion import DataIngestion
#     from src.router import Router

#     question = "What is agent memory?"

#     router = Router()
#     result = router.run(question)
#     print(f"Route: {result}")

#     if result["datasource"] == "vectorstore":
#         ingestion = DataIngestion()
#         retriever = ingestion.run()
#         docs = retriever.invoke(question)

#         generator = AnswerGenerator()
#         answer = generator.run(question, docs)
#         print(f"Answer: {answer}")
#     else:
#         print("Routed to web search")
