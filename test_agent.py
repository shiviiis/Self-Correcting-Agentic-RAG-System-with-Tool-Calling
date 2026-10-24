# run_tests.py
from dotenv import load_dotenv
load_dotenv()
from graph.build_graph import build_graph
from graph.state import initial_state

app = build_graph()


def ask(question, thread_id, max_retries=2):
    state =initial_state(question,max_retries)
    config = {"configurable": {"thread_id": thread_id}}

    path = []
    final = {}
    # stream_mode="updates" yields one item per node that ran
    for update in app.stream(state, config=config, stream_mode="updates"):
        for node_name, node_output in update.items():
            path.append(node_name)
            final.update(node_output or {})
    answer = final.get("generation","")
    print(f"\nQ: {question}")
    print("PATH:", " -> ".join(path))
    print("RETRIES:", path.count("rewrite_query"))
    print("CITED:", "[1]" in answer or "[2]" in answer)
    print("SOURCES:", [(s.get("source"), s.get("page") or s.get("url")) for s in final.get("doc_sources", [])][:3])
    print("A:", final.get("generation", "")[:300])
    return path


# Test 1: happy path (use a question that docs answer)
ask("How Microsoft was founded?", "t1")

# Test 2: fallback to web
ask("What is the current price of Bitcoin today?", "t2")

# Test 3: unanswerable anywhere, checks the retry limit
ask("What is the name of my neighbour's cat?", "t3")

ask("Who founded GitHub?", "chat1")
ask("And who acquired it?", "chat1")

ask("What was Microsoft's revenue in the year 1850?", "t5", max_retries=0)