# Reliable RAG Agent using LangGraph

A production-style **Retrieval-Augmented Generation (RAG)** agent that combines Adaptive RAG, Corrective RAG, and Self-RAG patterns into a single stateful pipeline. Built end-to-end with local, open-source tooling — no OpenAI required.

## Key Concepts Covered

- **Adaptive RAG** — dynamically routes questions to a vector store or live web search based on topic relevance
- **Corrective RAG** — grades retrieved documents for relevance and falls back to web search when chunks are irrelevant
- **Self-RAG** — uses LLM-as-judge to detect hallucinations and re-generate if the answer is not grounded in sources
- **LangGraph** — stateful graph orchestration with nodes, conditional edges, and automatic retries
- **ChromaDB** — local vector database with persistent storage and a REST API for inspecting embeddings
- **Pinecone** — cloud vector database used as an alternative to Chroma, with a visual dashboard for exploring stored vectors and similarity scores
- **LLM-as-Judge** — separate grader chains evaluate document relevance, hallucination, and answer quality using the same local LLM
- **RAGAs Evaluation** — end-to-end pipeline evaluation measuring faithfulness, answer relevancy, context precision, and context recall using local embeddings
- **Web Search fallback** — when retrieved chunks are irrelevant, the pipeline falls back to Tavily web search to supplement with live results
- **Full local isolation** — LLM (Ollama), embeddings (FastEmbed), and vector store (ChromaDB) all run locally with no external API calls, making the core pipeline fully offline-capable and cost-free

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.2 3B via Ollama (local) |
| Embeddings | BAAI/bge-base-en-v1.5 via FastEmbed (local) |
| Vector Store | ChromaDB (local) / Pinecone (cloud) |
| Orchestration | LangGraph + LangChain |
| Web Search | Tavily API |
| Evaluation | RAGAs |

## Pipeline

```
User Question
      ↓
Route Question ──────────────────────┐
      │ vectorstore                  │ web search
      ↓                              ↓
  Retrieve                       Web Search
      ↓                              │
Grade Documents ◄──────────────────┘
      ↓ (filter irrelevant chunks)
  Generate Answer
      ↓
Check Hallucination → not grounded → Generate (retry)
      ↓ grounded
  Grade Answer → not useful → Web Search
      ↓ useful
      END
```

## Project Structure

```
├── src/
│   ├── config.py        # URLs, model names, chunk settings
│   ├── models.py        # LLM and embedding model initialisation
│   ├── ingestion.py     # Data loading, chunking, vector store (skips if already indexed)
│   ├── router.py        # Routes question to vectorstore or web search
│   ├── graders.py       # RetrievalGrader, HallucinationGrader, AnswerGrader (OOP with base class)
│   ├── generator.py     # Generates final answer from retrieved context
│   ├── graph.py         # Node and Edge classes for LangGraph
│   └── workflow.py      # Builds and compiles the LangGraph pipeline
├── evaluation/
│   ├── evaluate.py      # RAGAs end-to-end evaluation script
│   └── requirements.txt
├── pinecone_db/
│   └── main.py          # Inserts chunks into Pinecone (cloud alternative to Chroma)
├── requirements.txt
└── .env                 # API keys (not committed)
```

## Setup

**1. Clone and create virtual environment:**
```bash
git clone <repo-url>
cd reliable-rag-langgraph
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Install Ollama and pull the model:**
```bash
# Install from https://ollama.com
ollama pull llama3.2:3b
```

**3. Set environment variables in `.env`:**
```
TAVILY_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here   # optional
```

**4. Start ChromaDB server:**
```bash
chroma run --path ./chroma_db
```

**5. Run ingestion (once — skips automatically on subsequent runs):**
```bash
python -m src.ingestion
```

**6. Run the RAG pipeline:**
```bash
python -m src.workflow
```

**7. Run RAGAs evaluation:**
```bash
pip install -r evaluation/requirements.txt
python evaluation/evaluate.py
```

## Evaluation Results (sample)

| Metric | Score | Description |
|---|---|---|
| `faithfulness` | 0.71 | Answer grounded in retrieved docs |
| `answer_relevancy` | 0.86 | Answer addresses the question |
| `context_precision` | 1.00 | Retrieved chunks are relevant |
| `context_recall` | 0.33 | Coverage of relevant information |

## Dataset

9 AI/ML blog posts from [Lilian Weng](https://lilianweng.github.io) covering LLM agents, prompt engineering, adversarial attacks, transformers, diffusion models, vision-language models, and training large models.
