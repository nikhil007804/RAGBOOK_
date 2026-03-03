import streamlit as st
import requests
import time
import json

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Agentic RAG Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main Background & Fonts */
    .stApp {
        background-color: #f8f9fc;
        color: #1a1e23;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Subtlest gradient on headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Input area adjustments */
    .stChatInputContainer {
        border-radius: 20px !important;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Custom Chat Bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s ease-in-out;
        border: 1px solid #f3f4f6;
        background-color: #ffffff;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="chatAvatarIcon-user"] {
        background-color: #3b82f6 !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #8b5cf6 !important;
    }

    /* Sidebar Tweaks */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .css-1lnzQJ9, .css-qbe2hs {
        color: #4b5563 !important;
    }

    /* Citation Expanders */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        color: #111827;
        border-radius: 8px;
        border: 1px solid #e5e7eb !important;
        font-size: 0.95em;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background-color: #ffffff;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding-left: 1rem;
        color: #374151;
        font-size: 0.95em;
    }
    
    /* Metrics / Status Tags */
    .status-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
        margin-right: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tag-high { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .tag-medium { background-color: #fef08a; color: #854d0e; border: 1px solid #fde047; }
    .tag-low { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .tag-cache { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    .tag-pipeline { background-color: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }

    /* Markdown Text Readability */
    p, li {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1f2937;
    }

    /* Keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# APPLICATION LOGIC
# ==========================================
API_URL = "http://localhost:8000"

def get_health_status():
    """Ping the FastAPI backend health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def query_backend(prompt: str, alpha: float):
    """Send query to the RAG backend."""
    try:
        payload = {"query": prompt, "alpha": alpha}
        response = requests.post(f"{API_URL}/query", json=payload, timeout=90)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Backend API Error: {str(e)}"}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the **Agentic Design Patterns** RAG Explorer. How can I assist you with analyzing the book?", "meta": None}
    ]

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ RAG Settings")
    st.markdown(
        "Control how the backend performs vector search. "
        "A lower alpha prioritizes exact keyword matching (BM25 sparse vectors), "
        "while a higher alpha prioritizes semantic intent (E5 dense vectors)."
    )
    
    alpha_slider = st.slider(
        "Hybrid Search Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="0.0 = Pure Keyword Search (BM25), 1.0 = Pure Semantic Search (Dense). Default = 0.6",
    )
    
    st.markdown("---")
    st.markdown("### 🔌 System Diagnostics")
    
    with st.spinner("Pinging services..."):
        health = get_health_status()
        time.sleep(0.5) # small delay for UI effect
        
        if health:
            st.success("🟢 API Server Connected")
            redis_color = "🟢" if health.get("redis") == "connected" else "🔴"
            pc_color = "🟢" if health.get("pinecone") == "connected" else "🔴"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Redis**  <br>{redis_color} {health.get('redis')}", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Pinecone**  <br>{pc_color} {health.get('pinecone')}", unsafe_allow_html=True)
        else:
            st.error("🔴 API Server Offline")
            st.markdown("Please make sure `uvicorn api.main:app` is running on port 8000.")

    st.markdown("---")
    st.caption("🚀 Powered by FastAPI, Pinecone E5, and Gemini 2.5 Flash")

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
st.title("Agentic RAG Explorer")
st.markdown("Ask conceptual questions about Agentic loops, extract code snippets, or find references to diagrams in the text.")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙎‍♂️" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])
        
        # If it's an assistant response with metadata
        if msg.get("meta"):
            meta = msg["meta"]
            
            # Draw pills
            conf_level = str(meta.get("confidence", "low")).lower()
            src_level = str(meta.get("source", "pipeline")).lower()
            st.markdown(f"""
            <div style="margin-top: 10px; margin-bottom: 15px;">
                <span class="status-tag tag-{conf_level}">Confidence: {conf_level}</span>
                <span class="status-tag tag-{src_level}">Source: {src_level}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Draw Citations
            citations = meta.get("citations", [])
            if citations and isinstance(citations, list) and len(citations) > 0:
                with st.expander(f"📚 View Citations ({len(citations)} found)"):
                    for i, cit in enumerate(citations):
                        pg = cit.get("page", "?")
                        ch = cit.get("chapter", "Unknown")
                        ex = cit.get("excerpt", "")
                        st.markdown(f"**[{i+1}] Page {pg} — {ch}**")
                        st.caption(f"> {ex}")
                        if i < len(citations) - 1:
                            st.divider()

# Handle new user input
if prompt := st.chat_input("Ask a question about the Agentic Design document..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user", avatar="🙎‍♂️"):
        st.markdown(prompt)

    # Trigger assistant response
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Retrieving context & generating answer..."):
            
            start_time = time.time()
            res = query_backend(prompt, alpha_slider)
            elapsed = time.time() - start_time
            
            if "error" in res:
                error_msg = f"❌ **Request Failed:**\n```\n{res['error']}\n```"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "meta": None})
            else:
                answer = res.get("answer", "No answer generated.")
                st.markdown(answer)
                
                # Metadata / pills
                conf_level = str(res.get("confidence", "low")).lower()
                src_level = str(res.get("source", "pipeline")).lower()
                query_type = str(res.get("query_type", "prose")).lower()
                
                st.markdown(f"""
                <div style="margin-top: 10px; margin-bottom: 15px;">
                    <span class="status-tag tag-{conf_level}">Confidence: {conf_level}</span>
                    <span class="status-tag tag-{src_level}">Source: {src_level} ({elapsed:.1f}s)</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Citations
                citations = res.get("citations", [])
                if citations and isinstance(citations, list) and len(citations) > 0:
                    with st.expander(f"📚 View Citations ({len(citations)} found)"):
                        for i, cit in enumerate(citations):
                            pg = cit.get("page", "?")
                            ch = cit.get("chapter", "Unknown")
                            ex = cit.get("excerpt", "")
                            st.markdown(f"**[{i+1}] Page {pg} — {ch}**")
                            st.caption(f"> {ex}")
                            if i < len(citations) - 1:
                                st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "meta": {
                        "confidence": conf_level,
                        "source": src_level,
                        "citations": citations
                    }
                })
