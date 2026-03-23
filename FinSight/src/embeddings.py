"""
src/embeddings.py — Phase 3a
Loads BAAI/bge-small-en-v1.5, embeds all chunks, saves vectors to disk.

Run from finsight/ root:
    python src/embeddings.py
"""

import os
import time
import numpy as np
from tqdm import tqdm

# Disable HuggingFace network access globally
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer


# Wrapper to make SentenceTransformer compatible with LangChain APIs
class EmbeddingAdapter:
    """Wraps SentenceTransformer to be compatible with LangChain Chroma/RAGAS."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: list) -> list:
        """LangChain API: embed list of documents."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list:
        """LangChain API: embed a single query."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


# ─────────────────────────────────────────────
# 1. LOAD THE EMBEDDING MODEL
# ─────────────────────────────────────────────

def load_embedding_model() -> EmbeddingAdapter:
    """
    Loads BAAI/bge-small-en-v1.5 from local HuggingFace cache only.
    Does NOT attempt network connection — avoids Windows Firewall issues.

    Why bge-small-en-v1.5?
      - Top-ranked on MTEB leaderboard for its size class
      - 384-dimensional vectors (compact but powerful)
      - Fast on CPU — no GPU needed
    """
    print("\n Loading embedding model: BAAI/bge-small-en-v1.5 (local cache only) ...")

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    try:
        # Load ONLY from local cache — no network access
        model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device="cpu",
            cache_folder=cache_dir,
            trust_remote_code=False,
            local_files_only=True,  # CRITICAL: Prevent all network access
        )
        print(f" [OK] Model loaded from local cache: {cache_dir}")
    except Exception as e:
        print(f" [ERROR] Failed to load from cache: {e}")
        print(f" Cache location: {cache_dir}")
        print(" Please ensure the model has been downloaded in previous phases.")
        print(" Run: python src/embeddings.py")
        raise
    
    test_vec = model.encode("test", normalize_embeddings=True)
    print(f" Embedding dimension: {len(test_vec)}")
    
    # Wrap model to be compatible with LangChain APIs
    adapter = EmbeddingAdapter(model)
    return adapter


# ─────────────────────────────────────────────
# 2. EMBED ALL CHUNKS IN BATCHES
# ─────────────────────────────────────────────

def embed_chunks(chunks: list, model: EmbeddingAdapter, batch_size: int = 32) -> np.ndarray:
    """
    Embeds all chunks in batches of batch_size.

    Returns numpy array of shape (num_chunks, 384).
    Batching keeps memory flat regardless of corpus size.
    """
    print(f"\n Embedding {len(chunks)} chunks in batches of {batch_size} ...")
    start = time.time()

    all_texts      = [chunk.page_content for chunk in chunks]
    batch_embeddings = model.model.encode(
        all_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - start
    vectors = np.array(batch_embeddings, dtype=np.float32)

    print(f" Embedded {len(chunks)} chunks in {elapsed:.1f}s")
    print(f" Embedding matrix shape: {vectors.shape}")
    return vectors


# ─────────────────────────────────────────────
# 3. SAVE & LOAD VECTORS
# ─────────────────────────────────────────────

def save_embeddings(vectors: np.ndarray, path: str = "data/embeddings.npy") -> None:
    """Saves embedding vectors to disk. Never re-embed unless corpus changes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, vectors)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f" Saved embeddings to {path} ({size_mb:.1f} MB)")

def load_embeddings(path: str = "data/embeddings.npy") -> np.ndarray:
    """Loads previously saved embedding vectors from disk."""
    vectors = np.load(path)
    print(f" Loaded embeddings from {path} — shape: {vectors.shape}")
    return vectors


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    CHUNKS_PATH     = "data/chunks.pkl"
    EMBEDDINGS_PATH = "data/embeddings.npy"

    print(" Loading chunks from Phase 2 ...")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f" Loaded {len(chunks)} chunks")

    model   = load_embedding_model()
    vectors = embed_chunks(chunks, model, batch_size=32)
    save_embeddings(vectors, EMBEDDINGS_PATH)

    print("\n Phase 3a complete. Run vectorstore.py next.")