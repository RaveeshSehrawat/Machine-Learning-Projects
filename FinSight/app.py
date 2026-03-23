"""
app.py — FinSight Streamlit Application (HuggingFace Spaces)

For HuggingFace Spaces deployment:
    Set repo_type=space and select Streamlit in your Space settings

Run locally:
    streamlit run app.py
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Global state ─────────────────────────────────────────────────────────────
pipeline = None
conversation_history = []

# ── Pipeline loader ──────────────────────────────────────────────────────────
def load_default_pipeline():
    """Load FinSight pipeline from pre-built dataset"""
    global pipeline
    if pipeline is not None:
        return  # Already loaded
    
    if not os.path.exists("chroma_db") or not os.path.exists("data/embeddings.npy"):
        error_msg = (
            "❌ **Setup required:** Pre-built data not found.\n\n"
            "Please run these setup commands in your terminal:\n\n"
            "```bash\n"
            "python src/ingest.py        # Process CSV data\n"
            "python src/embeddings.py    # Generate embeddings\n"
            "python src/vectorstore.py   # Build vector database\n"
            "```\n\n"
            "Then run Streamlit again: `streamlit run app.py`"
        )
        raise ValueError(error_msg)
    
    from pipeline import FinSightPipeline
    pipeline = FinSightPipeline()

# ── Query handler ────────────────────────────────────────────────────────────
def answer_question(question: str, strategy: str) -> tuple:
    """
    Process a question and return formatted answer with sources
    
    Returns:
        (answer_text, sources_markdown, chat_html)
    """
    global pipeline, conversation_history
    
    if not question.strip():
        return "", "❌ Please enter a question", ""
    
    try:
        # Load pipeline if not already loaded
        if pipeline is None:
            load_default_pipeline()
        
        # Query pipeline
        result = pipeline.query(question.strip(), strategy=strategy)
        
        answer = result["answer"]
        sources = result["sources"]
        strat = result["strategy_used"]
        
        # Format sources
        sources_md = f"**Strategy:** {strat}\n\n"
        if sources:
            sources_md += f"**Sources ({len(sources)} records):**\n\n"
            for i, src in enumerate(sources, 1):
                ticker = src.get("ticker", "N/A")
                year = src.get("year", "N/A")
                sentiment = src.get("sentiment_raw", "N/A")
                score = src.get("score", 0)
                gc_flag = " ⚠️ Going Concern" if src.get("going_concern") == "1" else ""
                text_preview = src.get("text", "")[:300] + "..."
                
                sources_md += (
                    f"{i}. **{ticker} ({year})**\n"
                    f"   - Relevance: {score:.3f}\n"
                    f"   - Sentiment: {sentiment}{gc_flag}\n"
                    f"   - {text_preview}\n\n"
                )
        else:
            sources_md += "❌ No relevant sources found. Try rephrasing with ticker + year."
        
        # Build conversation history for display
        conversation_history.append({"q": question, "a": answer, "strat": strat})
        chat_html = _build_chat_html()
        
        return answer, sources_md, chat_html
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        chat_html = _build_chat_html()
        return "", error_msg, chat_html

def _build_chat_html() -> str:
    """Format conversation history as HTML"""
    html = '<div style="max-height: 400px; overflow-y: auto; padding: 10px; background: #f5f5f5; border-radius: 8px;">'
    
    if not conversation_history:
        html += '<p style="color: #999; text-align: center;"><i>No conversation yet</i></p>'
    else:
        for turn in conversation_history:
            html += (
                f'<div style="margin-bottom: 12px; padding: 8px; background: white; border-radius: 4px;">'
                f'  <b>You:</b> {turn["q"]}<br>'
                f'  <b>FinSight ({turn["strat"]}):</b> {turn["a"][:200]}...'
                f'</div>'
            )
    
    html += '</div>'
    return html

# ── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="FinSight", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
# 📊 FinSight
**S&P 500 Financial Q&A powered by RAG**

Ask questions about S&P 500 companies — answers are cited directly from SEC 10-K financial data.
""")

st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.markdown("### How to use:")
    st.markdown("""
    1. Enter your question below
    2. Select retrieval strategy (hybrid recommended)
    3. Press "Ask FinSight"
    4. View answer with source citations
    """)
    
    st.markdown("### Example questions:")
    st.markdown("""
    - "What was GOOG's net margin in 2023?"
    - "Which companies had a going concern warning?"
    - "Did ETR's net income grow despite falling revenue in 2023?"
    - "What was BRO's revenue growth in 2025?"
    - "Which company had the most negative executive sentiment?"
    """)
    
    strategy = st.radio(
        "Retrieval Strategy",
        options=["hybrid", "dense", "sparse"],
        help="hybrid: BM25 + semantic + re-ranker (best) | dense: semantic only | sparse: keyword only"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Your Question")
    question_input = st.text_area(
        "Ask about S&P 500 companies",
        placeholder="e.g., What was GOOG's net margin in 2023?",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### ")
    st.markdown("")
    submit_btn = st.button("🔍 Ask FinSight", use_container_width=True, type="primary")

st.markdown("---")

# Results display
if submit_btn or "last_question" in st.session_state:
    if submit_btn and question_input.strip():
        st.session_state.last_question = question_input
        
        with st.spinner("Searching and generating answer..."):
            answer_text, sources_md, chat_html = answer_question(question_input, strategy)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Answer")
            st.markdown(answer_text if answer_text else "*Waiting for result...*")
        
        with col2:
            st.markdown("### Sources")
            st.markdown(sources_md)
    
    # Conversation history
    if "last_question" in st.session_state:
        st.markdown("---")
        st.markdown("### Conversation History")
        st.markdown(chat_html, unsafe_allow_html=True)

st.markdown("---")

with st.expander("📚 About FinSight"):
    st.markdown("""
    ### Technology Stack (100% free, runs locally)
    - **Embeddings:** BAAI/bge-small-en-v1.5 (384-dim)
    - **Vector DB:** ChromaDB (1,487 S&P 500 records)
    - **Retrieval:** BM25 keyword search + dense semantic search + RRF fusion
    - **Re-ranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
    - **LLM:** Mistral 7B via Ollama (local, no API key)
    
    ### Pipeline
    ```
    Question → BM25 top-20 + Dense top-20 → RRF fusion → top-10 
    → Cross-encoder re-rank → top-3 → Mistral generation → Answer
    ```
    
    ### Dataset: S&P 500 Alpha (Kaggle)
    - 1,487 company-year records
    - Financial ratios: Net Margin, ROA, Revenue Growth, Net Income Growth
    - NLP signals: Executive sentiment, MD&A complexity, going concern flags
    - Source: SEC EDGAR + XBRL APIs
    """)