"""
src/generator.py — Phase 5a

Loads Mistral via Ollama, builds the prompt template,
formats retrieved chunks into context, and generates answers.

Run standalone to test generation in isolation:
    python src/generator.py
"""

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────
# 1. LOAD THE LOCAL LLM (Mistral via Ollama)
# ─────────────────────────────────────────────

def load_llm(model_name: str = "mistral", temperature: float = 0.0) -> ChatOllama:
    """
    Loads Mistral running locally via Ollama.

    temperature=0.0 = deterministic output — critical for financial Q&A
    where you want consistent, factual answers, not creative variations.

    Ollama must be installed and running in the background.
    Test it works first: ollama run mistral "hello"
    """
    print(f"\n Loading LLM: {model_name} via Ollama (local, free, no API key) ...")
    llm = ChatOllama(model=model_name, temperature=temperature)
    print(f" LLM ready: {model_name}")
    return llm


# ─────────────────────────────────────────────
# 2. PROMPT TEMPLATE
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinSight, a financial data analyst assistant.

You have been given structured financial data records from SEC 10-K filings.
Each record contains exact financial metrics for a specific company and year.

CRITICAL RULES:
1. You MUST answer using the data in the context below. The data is already extracted and verified.
2. NEVER say "I cannot find" or "I don't have enough context" — the answer IS in the context.
3. Always state the specific number or value from the context in your answer.
4. Always end with: [Source: TICKER_YEAR]
5. Keep answers short and factual — one or two sentences maximum.

EXAMPLE:
Context: Company: CPB | Fiscal Year: 2024 ... Net Profit Margin: 5.88%
Question: What was CPB's net profit margin in 2024?
Answer: CPB reported a net profit margin of 5.88% in fiscal year 2024. [Source: CPB_2024]

Financial data:
──────────────────────────────────
{context}
──────────────────────────────────
"""


def build_prompt() -> ChatPromptTemplate:
    """Returns the ChatPromptTemplate used for every RAG query."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


# ─────────────────────────────────────────────
# 3. FORMAT CONTEXT FROM RETRIEVED CHUNKS
# ─────────────────────────────────────────────

def format_context(retrieved_chunks: list, max_chars: int = 3500) -> str:
    """
    Formats retrieved chunks into a clean context string for the prompt.

    Each chunk is labelled with its ticker and year so the LLM
    can cite it correctly in the answer.

    max_chars: rough limit to stay within Mistral's context window.
    At ~4 chars per token, 3500 chars ≈ 875 tokens of context.
    This leaves plenty of room for the system prompt + answer.
    """
    context_parts = []
    total_chars   = 0

    for i, result in enumerate(retrieved_chunks):
        doc     = result["doc"]
        ticker  = doc.metadata.get("ticker",  doc.metadata.get("company", "UNKNOWN")).upper()
        year    = doc.metadata.get("year",    "?")
        text    = doc.page_content.strip()

        chunk_str = f"[Source {i+1}: {ticker}_{year}]\n{text}\n"

        if total_chars + len(chunk_str) > max_chars:
            break

        context_parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n".join(context_parts)


# ─────────────────────────────────────────────
# 4. GENERATE AN ANSWER
# ─────────────────────────────────────────────

def generate_answer(
    question        : str,
    retrieved_chunks: list,
    llm             : ChatOllama,
    min_score       : float = 0.25,
) -> dict:
    """
    Generates a grounded answer from Mistral given a question + retrieved chunks.

    Steps:
        1. Guard: if no chunks or top chunk score is too low, return fallback
        2. Format retrieved chunks into a context string
        3. Build prompt and invoke Mistral
        4. Return answer string + structured sources list

    min_score: relevance threshold — only applied for dense retrieval
    (cosine similarity 0–1). RRF hybrid scores are tiny floats ~0.03,
    so the threshold is skipped for those.

    Returns:
        {
            "answer"             : str,
            "sources"            : list of dicts,
            "skipped_generation" : bool
        }
    """

    # ── Guard: no chunks at all ───────────────────────────────────────────
    if not retrieved_chunks:
        return {
            "answer"             : "I couldn't find relevant information in the loaded dataset.",
            "sources"            : [],
            "skipped_generation" : True,
        }

    # ── Guard: low relevance (dense retrieval only) ───────────────────────
    top_score  = retrieved_chunks[0].get("score", 1.0)
    retriever  = retrieved_chunks[0].get("retriever", "hybrid")
    if retriever == "dense" and top_score < min_score:
        return {
            "answer"             : (
                "I couldn't find relevant information for that question. "
                "Try rephrasing or asking about a specific company ticker and year."
            ),
            "sources"            : [],
            "skipped_generation" : True,
        }

    # ── Format context + build chain ─────────────────────────────────────
    context  = format_context(retrieved_chunks)
    prompt   = build_prompt()
    chain    = prompt | llm

    # ── Call Mistral ──────────────────────────────────────────────────────
    response = chain.invoke({
        "context" : context,
        "question": question,
    })
    answer = response.content if hasattr(response, "content") else str(response)

    # ── Build structured sources list for the UI ──────────────────────────
    sources = []
    for r in retrieved_chunks:
        doc    = r["doc"]
        ticker = doc.metadata.get("ticker", doc.metadata.get("company", "?")).upper()
        year   = doc.metadata.get("year", "?")
        sources.append({
            "ticker"         : ticker,
            "year"           : year,
            "filename"       : doc.metadata.get("filename", f"{ticker}_{year}_10K.txt"),
            "going_concern"  : doc.metadata.get("going_concern", "0"),
            "sentiment_raw"  : doc.metadata.get("sentiment_raw", "0"),
            "text"           : doc.page_content[:350].strip(),
            "score"          : round(r.get("score", 0), 4),
        })

    return {
        "answer"             : answer,
        "sources"            : sources,
        "skipped_generation" : False,
    }


# ─────────────────────────────────────────────
# MAIN — test generation without running full pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Mock a retrieved chunk using real data from the dataset
    mock_chunks = [
        {
            "doc": type("Doc", (), {
                "page_content": (
                    "Company: AAPL | Fiscal Year: 2023\n\n"
                    "--- Financial Performance ---\n"
                    "Net Profit Margin      : 25.31%\n"
                    "Return on Assets (ROA) : 28.29%\n"
                    "Revenue Growth YoY     : -2.80%\n"
                    "Net Income Growth YoY  : -2.81%\n\n"
                    "--- Executive Narrative (MD&A Analysis) ---\n"
                    "FinBERT Sentiment Score : Negative (-0.248)\n"
                    "Writing Complexity      : Grade 14.8 — Complex (college level)\n"
                    "Auditor Warning         : No going concern warning issued."
                ),
                "metadata": {
                    "ticker"        : "AAPL",
                    "year"          : "2023",
                    "going_concern" : "0",
                    "sentiment_raw" : "-0.2483",
                    "source"        : "AAPL_2023",
                    "filename"      : "AAPL_2023_10K.txt",
                    "chunk_id"      : "0",
                },
            })(),
            "score"    : 0.88,
            "chunk_id" : "0",
            "retriever": "dense",
        }
    ]

    llm    = load_llm()
    result = generate_answer("What was Apple's net profit margin in 2023?", mock_chunks, llm)

    print("\n Answer:")
    print(result["answer"])
    print("\n Sources used:")
    for s in result["sources"]:
        print(f"  - {s['ticker']} {s['year']} | sentiment: {s['sentiment_raw']} | score: {s['score']}")

    print("\n Phase 5a complete. Run pipeline.py next.")