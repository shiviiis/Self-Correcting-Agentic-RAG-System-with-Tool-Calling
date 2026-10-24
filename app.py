import re
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.build_graph import build_graph
from graph.state import initial_state
from ingestion.ingestion_pipeline import DOCS_DIR, ingest

st.set_page_config(page_title="Self-Correcting Agentic RAG", page_icon="🔎")
st.title("🔎 Self-Correcting Agentic RAG")

LABELS = {
    "condense_question": "Resolved follow-up using chat history",
    "check_question": "Checked the question is answerable",
    "retrieve_pdf": "Searched uploaded documents",
    "grade_documents": "Graded retrieved context",
    "web_search": "Fell back to live web search (Tavily)",
    "rewrite_query": "Rewrote the query and retried",
    "generate": "Generated answer",
    "grade_answer": "Graded the answer",
    "finalize": "Saved to memory",
}


@st.cache_resource          # build the graph once, not on every Streamlit rerun
def get_app():
    return build_graph()


app = get_app()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())   # one memory thread per browser session
    st.session_state.history = []                     # only for re-displaying the chat

#  Sidebar
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if files and st.button("Index documents"):
        DOCS_DIR.mkdir(exist_ok=True)
        for f in files:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Indexing..."):
            total = ingest()
        st.success(f"Indexed. {total} chunks in the database.")

    st.header("Settings")
    max_retries = st.slider("Max retries", 0, 3, 2)
    if st.button("New chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()


def show_sources(sources, answer):
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    rows = [(i, s) for i, s in enumerate(sources, 1) if not cited or i in cited]
    if not rows:
        return
    with st.expander(f"Sources ({len(rows)})"):
        for i, s in rows:
            if s["type"] == "web":
                st.markdown(f"**[{i}]** [{s['source']}]({s['url']})")
            else:
                page = f", page {s['page']}" if s.get("page") else ""
                st.markdown(f"**[{i}]** {s['source']}{page}")
            st.caption(s["content"][:300] + "...")


#  Replay chat 
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m["sources"], m["content"])

# New question 
if question := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    final, trace = {}, []

    with st.chat_message("assistant"):
        try:
            with st.status("Working...", expanded=True) as status:
                for update in app.stream(initial_state(question, max_retries), config=config, stream_mode="updates"):
                    for node, out in update.items():
                        trace.append(node)
                        st.write(f"✅ {LABELS.get(node, node)}")
                        final.update(out or {})
                status.update(label=f"Done ({trace.count('rewrite_query')} retries)", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Something went wrong (check your internet and API keys): {e}")
            st.stop()

        answer = final.get("generation", "")
        sources = final.get("doc_sources", [])
        st.markdown(answer)
        show_sources(sources, answer)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})