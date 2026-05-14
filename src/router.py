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
    

    def run(self):
        question_router = self.prompt | self.models.json_llm | JsonOutputParser()
        questions = ["transformers", "GPT", "Countries"]
        for question in questions:
            answer = question_router.invoke({"question": question})
        print(answer)

if __name__ == "__main__":
    r = Router()
    r.run()

