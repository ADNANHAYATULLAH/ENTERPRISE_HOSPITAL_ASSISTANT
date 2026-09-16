"""
Hospital Knowledge Base Assistant
Streamlit chat app: retrieves relevant chunks from a local FAISS index,
sends them to Groq (openai/gpt-oss-120b) to generate an answer,
and shows which source document(s) the answer came from.
"""

import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = "index_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful hospital knowledge base assistant. Answer the user's "
    "question using ONLY the information in the provided context excerpts. "
    "If the answer is not contained in the context, say you don't have that "
    "information in the knowledge base rather than guessing. Be concise and "
    "clear, and write for hospital staff."
)

DEPT_ICONS = {
    "education": "🎓",
    "patient admission": "🏥",
    "admission": "🏥",
    "engineering": "🔧",
    "facilities": "🔧",
    "it": "💻",
    "information technology": "💻",
    "hospital rules": "📋",
    "general": "📄",
}


def dept_icon(dept: str) -> str:
    if not dept:
        return "📄"
    key = dept.strip().lower()
    for k, icon in DEPT_ICONS.items():
        if k in key:
            return icon
    return "📄"


# ---------------------------------------------------------------------------
# Page config + custom styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ---- Page background ---- */
    .stApp {
        background: linear-gradient(180deg, #f4f8fb 0%, #eef3f8 100%);
    }

    /* ---- Hide default Streamlit chrome ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Header banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #0f6b62 0%, #14877a 45%, #1aa596 100%);
        border-radius: 18px;
        padding: 2rem 2rem 1.6rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(15, 107, 98, 0.25);
    }
    .hero-title {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 0.98rem;
        margin-top: 0.35rem;
        font-weight: 400;
    }
    .hero-badges {
        margin-top: 0.9rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255,255,255,0.16);
        color: #fff;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.25);
    }

    /* ---- Chat bubbles ---- */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.6rem;
        animation: fadeIn 0.25s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* User message bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #dff3ef;
        border: 1px solid #b9e4da;
    }

    /* Assistant message bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff;
        border: 1px solid #e3e9ee;
        box-shadow: 0 2px 8px rgba(20, 40, 60, 0.04);
    }

    /* ---- Sources expander ---- */
    details {
        background: #f6faf9;
        border: 1px solid #d9ece8;
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
        margin-top: 0.3rem;
    }
    details summary {
        font-weight: 600;
        color: #0f6b62;
        cursor: pointer;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #0f2f2b;
    }
    [data-testid="stSidebar"] * {
        color: #eef7f4 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }
    .sidebar-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.9rem;
        color: #7fe0cd !important;
    }
    .sidebar-card p {
        font-size: 0.85rem;
        line-height: 1.4;
        margin: 0;
        opacity: 0.9;
    }

    /* ---- Sidebar button ---- */
    [data-testid="stSidebar"] button {
        background: #14877a !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #1aa596 !important;
    }

    /* ---- Chat input box ---- */
    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* ---- Spinner text ---- */
    .stSpinner > div {
        color: #0f6b62;
        font-weight: 500;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🏥 Hospital Knowledge Base Assistant</p>
        <p class="hero-subtitle">Ask about policies, procedures, and department guidelines — answers are grounded in your official documents.</p>
        <div class="hero-badges">
            <span class="hero-badge">📋 Hospital Rules</span>
            <span class="hero-badge">🏥 Patient Admission</span>
            <span class="hero-badge">🔧 Engineering</span>
            <span class="hero-badge">💻 IT</span>
            <span class="hero-badge">🎓 Education</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources (loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found in Streamlit secrets. "
            "Add it to .streamlit/secrets.toml or your app's Secrets settings."
        )
        st.stop()
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def retrieve_chunks(vectorstore, question, k=TOP_K):
    results = vectorstore.similarity_search(question, k=k)
    return results


def build_context(chunks):
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        dept = chunk.metadata.get("department", "General")
        source = chunk.metadata.get("source_file", "Unknown document")
        page = chunk.metadata.get("page")
        page_str = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(
            f"[Excerpt {i} — {dept} / {source}{page_str}]\n{chunk.page_content}"
        )
    return "\n\n".join(blocks)


def generate_answer(client, question, context, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # include recent chat history for conversational context
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append(
        {
            "role": "user",
            "content": f"Context excerpts:\n\n{context}\n\nQuestion: {question}",
        }
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def unique_sources(chunks):
    seen = []
    for chunk in chunks:
        dept = chunk.metadata.get("department", "General")
        source = chunk.metadata.get("source_file", "Unknown document")
        page = chunk.metadata.get("page")
        page_str = f" (page {page + 1})" if isinstance(page, int) else ""
        icon = dept_icon(dept)
        label = f"{icon} **{source}**{page_str} — _{dept}_"
        if label not in seen:
            seen.append(label)
    return seen


# ---------------------------------------------------------------------------
# App state + resources
# ---------------------------------------------------------------------------
vectorstore = load_vectorstore()
client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 About")
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>What this does</h4>
            <p>Answers questions using your hospital's internal policy
            documents. Every answer is grounded in retrieved excerpts —
            never made up.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Covered departments</h4>
            <p>🎓 Education &nbsp;•&nbsp; 🏥 Patient Admission<br>
            🔧 Engineering &nbsp;•&nbsp; 💻 IT<br>
            📋 Hospital Rules</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 1rem; color: #5a6b70;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">💬</div>
            <div style="font-size: 1rem; font-weight: 600; color:#0f6b62;">Ask your first question</div>
            <div style="font-size: 0.88rem; margin-top: 0.3rem;">e.g. "What is the visiting hours policy?" or "How do I report a medical gas alarm?"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "user" else "🏥"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about hospital policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(vectorstore, question)
            context = build_context(chunks)
            sources = unique_sources(chunks)

        with st.spinner("Generating answer..."):
            answer = generate_answer(client, question, context, st.session_state.messages)

        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
