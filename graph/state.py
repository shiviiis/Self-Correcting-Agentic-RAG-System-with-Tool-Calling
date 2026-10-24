from typing import List, Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


def initial_state(question: str, max_retries: int = 2) -> dict:
    """Fresh per-turn values. Passed on every call, which resets everything
    except `messages` (the checkpointer keeps memory because of add_messages)."""
    return {
        "messages": [],
        "question": question,
        "search_query": question,
        "documents": [],
        "doc_sources": [],
        "source_type": "none",
        "retry_count": 0,
        "max_retries": max_retries,
        "doc_grade": "",
        "answer_grade": "",
        "generation": "",
    }
    

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    
    question: str
    search_query: str
    
    documents: List[str]
    doc_sources: List[dict]
    source_type: Literal["pdf","web","none"]
    
    retry_count: int
    max_retries: int
    doc_grade: Literal["relevant","irrelevant",""]
    answer_grade: Literal["grounded","not_grounded", ""]
    
    generation: str