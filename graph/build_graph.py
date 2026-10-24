from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.nodes import retrieve_pdf, web_search, grade_documents, rewrite_query, generate, grade_answer, check_question,condense_question,finalize
from graph.edges import decide_after_doc_grade, decide_after_answer_grade,decide_after_check


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_pdf", retrieve_pdf)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_answer", grade_answer)
    workflow.add_node("check_question",check_question)
    workflow.add_node("condense_question", condense_question)
    workflow.add_node("finalize", finalize)
    workflow.add_edge(START,"condense_question")
    workflow.add_edge("condense_question","check_question")
    workflow.add_conditional_edges(
        "check_question",
        decide_after_check,
        {"end": END, "retrieve_pdf": "retrieve_pdf"},
    )    
    workflow.add_edge("retrieve_pdf", "grade_documents")
    workflow.add_edge("web_search", "grade_documents")   # web results get graded too

    # ONE conditional edge for grade_documents, with all three possible targets
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_doc_grade,
        {
            "generate": "generate",
            "web_search": "web_search",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("generate", "grade_answer")
    workflow.add_edge("rewrite_query", "retrieve_pdf")
    workflow.add_conditional_edges(
    "grade_answer",
    decide_after_answer_grade,
    {"end": "finalize", "rewrite_query": "rewrite_query"},
    )
    
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=MemorySaver())