# Self-Correcting Agentic RAG System

An agentic RAG pipeline that retrieves from uploaded documents, falls back to live
web search, grades its own context and answers, and retries with rewritten queries.

## Demo

### Cited answer from documents
![Cited answer](assets/answer-with-citations.png)

### Private-question guard
![Agent trace](assets/refusal.png)

### Web fallback with sources
![Web fallback](assets/web-fallback.png)

### Follow-up
![Follow-up and refusal](assets/follow-up-questions.png)

## Features
- PDF/TXT semantic retrieval (Chroma + sentence-transformers) with incremental ingestion
- Live web search fallback via Tavily
- LLM-as-judge grading of retrieved context and generated answers
- Query rewriting and bounded retries (configurable max)
- Conversational memory (LangGraph checkpointer) with follow-up question handling
- Inline citations with a sources panel
- Streamlit UI showing each step the agent takes

## Architecture

```mermaid
graph TD;
    __start__([Start]):::first
    condense_question(condense_question)
    check_question(check_question)
    retrieve_pdf(retrieve_pdf)
    grade_documents(grade_documents)
    web_search(web_search)
    rewrite_query(rewrite_query)
    generate(generate)
    grade_answer(grade_answer)
    finalize(finalize)
    __end__([End]):::last

    __start__ --> condense_question;
    condense_question --> check_question;
    check_question -. end .-> __end__;
    check_question -.-> retrieve_pdf;
    retrieve_pdf --> grade_documents;
    web_search --> grade_documents;
    grade_documents -.-> generate;
    grade_documents -.-> web_search;
    grade_documents -.-> rewrite_query;
    generate --> grade_answer;
    grade_answer -.-> rewrite_query;
    grade_answer -. end .-> finalize;
    rewrite_query --> retrieve_pdf;
    finalize --> __end__;

    classDef default fill:#e8f4fd,stroke:#1f6feb,stroke-width:1.5px,color:#0b2a4a
    classDef first fill:#d4f8d4,stroke:#2da44e,color:#0b3d1a
    classDef last fill:#ffd9d9,stroke:#cf222e,color:#5c0a12

    classDef retrieval fill:#dbeafe,stroke:#2563eb,color:#0b2a4a
    classDef grading fill:#fef3c7,stroke:#d97706,color:#4a2c00
    classDef generation fill:#dcfce7,stroke:#16a34a,color:#0b3d1a
    classDef retry fill:#fee2e2,stroke:#dc2626,color:#5c0a12
```
## Project structure

```
Self-Correcting-Agentic-RAG-System/
├── app.py                     # Streamlit UI
├── graph/
│   ├── state.py               # GraphState schema
│   ├── grading.py             # LLM graders, rewriter, question check
│   ├── nodes.py               # one function per graph node
│   ├── edges.py               # conditional routing logic
│   └── build_graph.py         # assembles the LangGraph workflow
├── tools/
│   ├── retriever_tool.py      # PDF/TXT retrieval
│   └── web_search_tool.py     # Tavily web search
├── ingestion/
│   └── ingestion_pipeline.py  # incremental indexing into Chroma
├── docs/                      # source documents
├── run_tests.py               # scenario tests
├── requirements.txt
└── .env.example
```

## Setup
```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in your keys
python -m ingestion.ingestion_pipeline
streamlit run app.py
```


## Known limitations
- Graders are LLM-based and can be fooled by plausible but irrelevant text.
- Live-data answers (prices) vary by source.
- Retrieval quality depends on chunking and the embedding model.
