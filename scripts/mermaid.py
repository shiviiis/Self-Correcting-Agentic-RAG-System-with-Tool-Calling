from dotenv import load_dotenv
load_dotenv()
from graph.build_graph import build_graph
print(build_graph().get_graph().draw_mermaid())