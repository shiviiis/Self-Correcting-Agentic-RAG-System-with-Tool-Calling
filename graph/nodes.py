from graph.state import GraphState
from graph.grading import doc_grader, answer_grader, query_rewriter, question_checker
from tools.retriever_tool import retrieve_chunks
from tools.web_search_tool import web_chunks
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from datetime import date
generation_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def _number(chunks: list[dict]) -> list[str]:
    """['[1] text...', '[2] text...'] so the LLM can cite by number."""
    return [f"[{i}] {c['content']}" for i, c in enumerate(chunks, 1)]

def retrieve_pdf(state: GraphState) -> dict:
    """Node: search internal documents using the current search_query."""
    print(f"---RETRIEVE PDF: {state['search_query']}---")
    chunks = retrieve_chunks(state["search_query"])
    return {
        "documents": _number(chunks),
        "doc_sources":chunks,
        "source_type": "pdf",
    }


def web_search(state: GraphState) -> dict:
    """Node: search the live web. Used when PDF retrieval was graded irrelevant."""
    print(f"---WEB SEARCH: {state['search_query']}---")
    chunks = web_chunks(state["search_query"])
    return {
        "documents": _number(chunks),
        "doc_sources": chunks,
        "source_type": "web",
    }


def grade_documents(state: GraphState) -> dict:
    """Node: LLM judges if retrieved documents actually help answer the question."""
    print("---GRADE DOCUMENTS---")
    docs_text = "\n\n".join(state["documents"])
    grade = doc_grader.invoke({
        "documents": docs_text,
        "question": state["question"],
    })
    print(f"   grade: {grade.binary_score}")
    
    if grade.binary_score.lower() == "yes":
        return {"doc_grade":"relevant"}
    
    update = {"doc_grade":"irrelevant"}
    if state["source_type"] == "web":
        update["documents"] = []
        update["doc_sources"] = []
    return update


def rewrite_query(state: GraphState) -> dict:
    """Node: reformulate the search query before retrying retrieval.Keep every constraint from the original question (such as 'my', names, dates, locations).
    Do not add new details. Do not make the question more generic."""
    print("---REWRITE QUERY---")
    result = query_rewriter.invoke({
        "question": state["question"],
        "search_query": state["search_query"],
        "today": date.today().isoformat(),
    })
    print(f"   new query: {result.rewritten_query}")
    return {
        "search_query": result.rewritten_query,
        "retry_count": state["retry_count"] + 1,
    }


def generate(state: GraphState) -> dict:
    """Node: produce the final answer using whatever documents are currently in state."""
    print("---GENERATE---")
    docs_text = "\n\n".join(state["documents"]) if state["documents"] else "No documents retrieved."

    prompt = f"""Answer the question using only the numbered sources below.
Rules:
- Cite sources inline like [1] or [2] after the claims they support.
- If sources give different values (e.g. live prices), give the range, say which source reported what, and note that it changes over time.
- If the sources do not clearly address the question, say "I don't have enough information to answer that question."
- Call them "sources", not "documents".
    
    Sources:
    {docs_text}
    
    Question: {state['question']}"""

    messages =([SystemMessage(content="You are a helpful assistant.")] + state["messages"][-6:] + [HumanMessage(content=prompt)])
    result = generation_model.invoke(messages)

    return {
        "generation": result.content
        # "messages": [HumanMessage(content=state["question"]), AIMessage(content=result.content)],
    }

REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "cannot find",
    "can't find",
]

def grade_answer(state: GraphState) -> dict:
    """Node: check if the generated answer is grounded and actually addresses the question."""
    print("---GRADE ANSWER---")
    answer_text = state["generation"].lower()
    if any(phrase in answer_text for phrase in REFUSAL_PHRASES):
        print("   answer is a refusal -> not_grounded")
        return {"answer_grade": "not_grounded"}
    docs_text = "\n\n".join(state["documents"]) if state["documents"] else "No documents."
    grade = answer_grader.invoke({
        "documents": docs_text,
        "question": state["question"],
        "answer": state["generation"],
    })
    print(f"   grounded: {grade.grounded}, addresses_question: {grade.addresses_question}")

    if grade.grounded.lower() == "yes" and grade.addresses_question.lower() == "yes":
        return {"answer_grade": "grounded"}
    return {"answer_grade": "not_grounded"}

def check_question(state: GraphState)-> dict:
    """Node : stop early if the question is about private information"""
    print("---CHECK QUESTION---")
    result = question_checker.invoke({"question":state["question"]})
    print(f"   answerable_publicly: {result.answerable_publicly}")

    if result.answerable_publicly.lower() == "no":
        msg = "That looks like personal or private information, so I can't look it up in documents or on the web."
        return {
            "generation": msg,
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=msg)],
            "doc_grade": "",
        }
    return {"generation":""}

def condense_question(state: GraphState) -> dict:
    """Turn 'And who acquired it?' into a standalone question using chat history."""
    print("---CONDENSE QUESTION---")
    history = state["messages"][-6:]
    if not history:                              # first turn, nothing to resolve
        return {"search_query": state["question"]}

    messages = (
        [SystemMessage(content=(
            "Rewrite the new question as a standalone question using the chat history. "
            "Keep its meaning. If it is already standalone, return it unchanged. "
            "Return only the question."))]
        + history
        + [HumanMessage(content=f"New question: {state['question']}")]
    )
    standalone = generation_model.invoke(messages).content.strip()
    print(f"   standalone: {standalone}")
    return {"question": standalone, "search_query": standalone}


def finalize(state: GraphState) -> dict:
    """Save the finished turn to memory exactly once."""
    return {"messages": [HumanMessage(content=state["question"]), AIMessage(content=state["generation"])]}
    