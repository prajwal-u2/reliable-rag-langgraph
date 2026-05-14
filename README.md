# Reliable RAG Agent using LangGraph

A production-style Retrieval-Augmented Generation (RAG) agent built with LangGraph, Ollama (Llama 3.1), and ChromaDB. Implements Adaptive RAG, Corrective RAG, and Self-RAG patterns.

## Architecture

- **LLM**: Llama 3.1 8B via Ollama (local, free)
- **Embeddings**: BAAI/bge-base-en-v1.5 via FastEmbed (local)
- **Vector Store**: ChromaDB (local server)
- **Framework**: LangChain + LangGraph
- **Web Search**: Tavily API (fallback retrieval)

## Project Structure

```
├── src/
│   ├── config.py       # URLs, model names, chunk settings
│   ├── models.py       # LLM and embedding model classes
│   └── ingestion.py    # Data loading, chunking, vector store
├── pinecone_db/
│   └── main.py         # Optional: insert chunks into Pinecone for visualization
├── requirements.txt
└── .env                # API keys (not committed)
```

## Setup

**1. Clone and create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Install and start Ollama:**
```bash
# Install from https://ollama.com
ollama pull llama3.1:8b
```

**3. Set environment variables in `.env`:**
```
TAVILY_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here   # optional, for visualization only
```

**4. Start ChromaDB server:**
```bash
chroma run --path ./chroma_db
```

**5. Run ingestion:**
```bash
python -m src.ingestion
```

## Datasets

8 AI/ML blog posts from Lilian Weng covering:
- LLM Agents
- Prompt Engineering
- Adversarial Attacks on LLMs
- Transformer Family
- Diffusion Models
- Vision-Language Models
- Training Large Models
