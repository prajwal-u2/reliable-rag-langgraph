URLS = [
    # Agents & reasoning
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    # Transformers & attention
    "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
    "https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/",
    # Generative models
    "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
    # Vision-language
    "https://lilianweng.github.io/posts/2022-06-09-vlm/",
    # Training large models
    "https://lilianweng.github.io/posts/2021-09-25-train-large/",
]

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3.1:8b"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 0
COLLECTION_NAME = "rag-chroma"
CHROMA_PERSIST_DIR = "./chroma_db"
