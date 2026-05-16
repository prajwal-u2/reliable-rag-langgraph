from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models import Models

class Router:

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at routing a user question to a vectorstore or web search.
                Use the vectorstore for questions on any of these topics:
                - LLM agents and autonomous agents
                - Prompt engineering techniques
                - Adversarial attacks on LLMs
                - Transformer architecture and attention mechanisms
                - Diffusion models and generative models
                - Vision-language models (VLMs)
                - Training large language models
                You do not need to be stringent with the keywords related to these topics.
                Otherwise, use web search.
                Return JSON with a single key 'datasource' with value 'web_search' or 'vectorstore'."""),
            ("human", "{question}"),
        ])
        self.models = Models()
    

    def run(self, question: str) -> dict:
        question_router = self.prompt | self.models.json_llm | JsonOutputParser()
        return question_router.invoke({"question": question})

# Testing only
# if __name__ == "__main__":
#     r = Router()
#     test_questions = ["What is agent memory?", "Who won the 2024 election?", "How do diffusion models work?"]
#     for q in test_questions:
#         print(f"{q} → {r.run(q)}")

