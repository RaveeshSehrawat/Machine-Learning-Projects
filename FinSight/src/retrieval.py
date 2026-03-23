"""
src/retrieval.py — Phase 4
Three retrieval strategies: dense, sparse (BM25), hybrid (RRF + reranker).

Run standalone to compare all three strategies:
    python src/retrieval.py
"""
import pickle
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_chroma import Chroma

_reranker = None

def _get_reranker() -> CrossEncoder:
    """Lazy-loads cross-encoder once and caches it. ~85MB cached locally."""
    global _reranker
    if _reranker is None:
        print(" Loading re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
        try:
            # Load with explicit device placement to avoid meta tensor errors
            import torch
            _reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu",
            )
            # Ensure model is on CPU (not in meta device mode)
            _reranker.model = _reranker.model.to("cpu")
            print(" Re-ranker loaded (CPU mode).")
        except Exception as e:
            import traceback
            print(f" [WARN] Failed to load re-ranker: {type(e).__name__}: {str(e)[:100]}")
            print(f" Re-ranking will be skipped — using hybrid without re-ranking.")
            _reranker = None  # Signal failure
    return _reranker


# ─────────────────────────────────────────────
# 1. DENSE RETRIEVAL
# ─────────────────────────────────────────────

def dense_retrieve(query: str, vectorstore: Chroma, k: int = 10) -> list:
    """
    Embeds query, finds k most similar chunks by cosine similarity.
    Best at: semantic questions, paraphrasing, conceptual queries.
    Weaker at: exact ticker symbols, specific years, keyword matching.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "doc"      : doc,
            "score"    : float(score),
            "chunk_id" : doc.metadata.get("chunk_id", str(i)),
            "retriever": "dense",
        }
        for i, (doc, score) in enumerate(results)
    ]


# ─────────────────────────────────────────────
# 2. BM25 SPARSE RETRIEVAL
# ─────────────────────────────────────────────

def build_bm25_index(chunks: list) -> BM25Okapi:
    """
    Builds BM25 keyword index over all chunks. Call once at startup.
    Best at: exact terms — ticker symbols (AAPL), years (2023),
             specific ratios, going concern, financial keywords.
    """
    print(f" Building BM25 index over {len(chunks)} chunks ...")
    tokenised = [c.page_content.lower().split() for c in chunks]
    index     = BM25Okapi(tokenised)
    print(" BM25 index built.")
    return index

def sparse_retrieve(query: str, chunks: list, bm25_index: BM25Okapi, k: int = 10) -> list:
    """Retrieves top-k chunks using BM25 keyword scoring."""
    scores      = bm25_index.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "doc"      : chunks[i],
            "score"    : float(scores[i]),
            "chunk_id" : chunks[i].metadata.get("chunk_id", str(i)),
            "retriever": "sparse",
        }
        for i in top_indices
    ]


# ─────────────────────────────────────────────
# 3. HYBRID — RRF FUSION
# ─────────────────────────────────────────────

def hybrid_retrieve(query: str, vectorstore: Chroma, chunks: list,
                    bm25_index: BM25Okapi, k: int = 10, rrf_k: int = 60) -> list:
    """
    Combines dense + sparse using Reciprocal Rank Fusion (RRF).
    score = 1/(rank_dense + 60) + 1/(rank_sparse + 60)
    rrf_k=60 is the standard constant — do not change it.
    Consistently outperforms either method alone on financial queries.
    """
    dense_results  = dense_retrieve(query, vectorstore, k=20)
    sparse_results = sparse_retrieve(query, chunks, bm25_index, k=20)

    rrf_scores   = defaultdict(float)
    chunk_lookup = {}

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        rrf_scores[cid]   += 1.0 / (rank + rrf_k)
        chunk_lookup[cid]  = r["doc"]

    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        rrf_scores[cid]   += 1.0 / (rank + rrf_k)
        chunk_lookup[cid]  = r["doc"]

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]
    return [
        {
            "doc"      : chunk_lookup[cid],
            "score"    : rrf_scores[cid],
            "chunk_id" : cid,
            "retriever": "hybrid",
        }
        for cid in sorted_ids
    ]


# ─────────────────────────────────────────────
# 4. CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────

def rerank(query: str, candidates: list, top_n: int = 3) -> list:
    """
    Re-scores top-10 hybrid candidates using cross-encoder.
    Cross-encoder reads (query, chunk) together — far more accurate
    than bi-encoders. Only run on top-10, not the full index.
    Pipeline: hybrid top-10 → reranker → top-3.
    
    Falls back to top-3 without re-ranking if re-ranker unavailable.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    
    # If reranker failed to load or not available, return top candidates without reranking
    if reranker is None:
        return candidates[:top_n]
    
    try:
        pairs    = [[query, c["doc"].page_content] for c in candidates]
        scores   = reranker.predict(pairs)
        
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
    except Exception as e:
        # Graceful fallback if prediction fails
        print(f" [WARN] Re-ranking failed ({type(e).__name__}), returning top-{top_n} without re-ranking")
        return candidates[:top_n]


# ─────────────────────────────────────────────
# 5. MASTER RETRIEVE FUNCTION
# ─────────────────────────────────────────────

def retrieve(query: str, vectorstore: Chroma, chunks: list,
             bm25_index: BM25Okapi, strategy: str = "hybrid", final_k: int = 3) -> list:
    """
    Single entry point for all retrieval strategies.
    Called by pipeline.py and app.py.

    strategy: "dense" | "sparse" | "hybrid"  (hybrid recommended)
    Returns final_k chunks as list of dicts.
    """
    if strategy == "dense":
        candidates = dense_retrieve(query, vectorstore, k=10)
    elif strategy == "sparse":
        candidates = sparse_retrieve(query, chunks, bm25_index, k=10)
    elif strategy == "hybrid":
        candidates = hybrid_retrieve(query, vectorstore, chunks, bm25_index, k=10)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: dense | sparse | hybrid")

    if strategy == "hybrid":
        return rerank(query, candidates, top_n=final_k)
    return candidates[:final_k]


# ─────────────────────────────────────────────
# MAIN — compare all three strategies
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from embeddings import load_embedding_model
    from vectorstore import load_vectorstore

    CHUNKS_PATH = "data/chunks.pkl"
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    model       = load_embedding_model()
    vectorstore = load_vectorstore(model)
    bm25_index  = build_bm25_index(chunks)

    test_queries = [
        "What was GOOG's net margin in 2023?",
        "Which companies had a going concern warning?",
        "What was CPB's executive sentiment in 2024?",
    ]

    for query in test_queries:
        print(f"\n{'═'*65}\n Query: {query}\n{'═'*65}")
        for strategy in ["dense", "sparse", "hybrid"]:
            results = retrieve(query, vectorstore, chunks, bm25_index, strategy=strategy)
            top = results[0]
            print(f"\n  [{strategy.upper()}] ticker={top['doc'].metadata.get('ticker','?')} | "
                  f"year={top['doc'].metadata.get('year','?')} | score={top['score']:.4f}")
            print(f"  {top['doc'].page_content[:150].strip()} ...")

    print("\n Phase 4 complete. Run generator.py next.")