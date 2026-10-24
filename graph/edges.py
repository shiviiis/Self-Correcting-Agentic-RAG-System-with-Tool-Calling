from graph.state import GraphState

def decide_after_doc_grade(state: GraphState) -> str:
    """After grading PDF docs: go straight to generate if relevant,
    otherwise fall back to web search — but only if retries remain."""
    if state["doc_grade"] == "relevant":
        return "generate"
    
    if state["source_type"] == "pdf":
        return "web_search"
    
    if state["retry_count"] < state["max_retries"]:
        return "rewrite_query"
    # retries exhausted, generate with whatever we have rather than loop forever
    return "generate"


def decide_after_answer_grade(state: GraphState) -> str:
    """After grading the final answer: end if good, otherwise retry
    (rewrite query -> retrieve again) up to max_retries."""
    if state["answer_grade"] == "grounded":
        return "end"
    if state["retry_count"] < state["max_retries"]:
        return "rewrite_query"
    return "end"  # give up gracefully, return best-effort answer

def decide_after_check(state:GraphState) -> str:
    #if check_question wrote a refusal into generation, we're done
    if state["generation"]:
        return "end"
    return "retrieve_pdf"