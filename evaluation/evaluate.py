import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics.collections import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from src.workflow import RagGraph
from src.models import Models

eval_data = {
    "question": [
        "What is agent memory?",
        "What are diffusion models?",
        "What is chain of thought prompting?",
        "When is the Independence day of USA",
        "Who is Narendra Modi?",
    ],
    "ground_truth": [
        "Agent memory allows LLMs to retain and recall information over time using external vector stores.",
        "Diffusion models are generative models that learn to reverse a noising process to generate data.",
        "Chain of thought prompting encourages models to reason step by step before answering complex tasks.",
        "The Independence Day of the USA is on July 4th, celebrated annually since 1776.",
        "Narendra Modi is the Prime Minister of India, serving since 2014, and leader of the BJP party.",
    ],
    "contexts": [],
    "answer": [],
}

eval_data = {
    "question": [
        "What is agent memory?",
        "Who is Narendra Modi?",
    ],
    "ground_truth": [
        "Agent memory allows LLMs to retain and recall information over time using external vector stores.",
        "Narendra Modi is the Prime Minister of India, serving since 2014, and leader of the BJP party.",
    ],
    "contexts": [],
    "answer": [],
}

print("Building RAG pipeline...")
app = RagGraph().build()

print("Running pipeline for each question...")
for question in eval_data["question"]:
    print(f"\nQ: {question}")
    result = app.invoke({"question": question})
    eval_data["answer"].append(result["generation"])
    eval_data["contexts"].append([d.page_content for d in result["documents"]])
    print(f"A: {result['generation'][:100]}...")

print("\nBuilding evaluation dataset...")
dataset = Dataset.from_dict(eval_data)

models = Models()
ragas_llm = LangchainLLMWrapper(models.json_llm)
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

print("\nRunning RAGAs evaluation...")
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(max_workers=1, timeout=180),
)

print()
print("--- RAGAs Evaluation Results ---")
print(results)
print(results.to_pandas())
