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
        label = f"{source}{page_str} — {dept}"
        if label not in seen:
            seen.append(label)
    return seen


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Hospital Knowledge Assistant", page_icon="🏥", layout="centered")

st.title("🏥 Hospital Knowledge Base Assistant")
st.caption("Ask a question about hospital policies, procedures, or department rules.")

vectorstore = load_vectorstore()
client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Chat input
question = st.chat_input("Ask a question about hospital policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
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

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This assistant answers questions using your hospital's internal "
        "policy documents. Answers are generated from retrieved excerpts only."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
