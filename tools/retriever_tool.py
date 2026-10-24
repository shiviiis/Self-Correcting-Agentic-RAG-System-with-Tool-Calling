from langchain_core.tools import tool
from  ingestion.ingestion_pipeline import get_vectorstore

_retriever = get_vectorstore().as_retriever(search_kwargs = {"k":5})

def retrieve_chunks(query : str)-> list[dict]:
    docs = _retriever.invoke(query)
    return [
        {
            "type":"pdf",
            "content":d.page_content,
            "source":d.metadata.get("source","unknown"),
            "page":(d.metadata["page"]+1) if "page" in d.metadata else None,
        }
        for d in docs
    ]

@tool
def search_pdf(query):
    """Search the internal company documents, including historical records, acquisition details, financial reports, and policies
    that have already been uploaded to the knowledge base. ALWAYS try this tool first for any
    factual question, even if it sounds like general knowledge - the answer may already be documented internally."""
    chunks = retrieve_chunks(query)
    if not chunks :
        return "No relevant documents found."
    formatted = []
    return "\n\n".join(
        f"[Source: {c['source']}, page {c['page'] or 'n/a'}]\n{c['content']}" for c in chunks
    )


    