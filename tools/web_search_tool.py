from langchain_core.tools import tool
from langchain_tavily import TavilySearch

_tavily = TavilySearch(max_results=5)


def web_chunks(query: str) -> list[dict]:
    results = _tavily.invoke(query)
    if isinstance(results,dict):
        results = results.get("results",[])
        
    return [
        {"type":"web", "content":r.get("content",""), "source":r.get("title",""),"url":r.get("url","")}
        for r in results
    ]
    
@tool
def search_web(query):
    """Search the live web. Use this ONLY if search_pdf has already been tried 
    and returned no relevant information, or if the question is explicitly 
    about something very recent (today's news, current stock price, this week's events)."""    
    results = web_chunks(query)
    if isinstance(results, dict):
        results = results.get("results", [])
    formatted = []
    for r in results:
        formatted.append(f"[{r.get('title','')}]({r.get('url','')})\n{r.get('content','')}")
    return "\n\n".join(formatted) if formatted else "No web results found"