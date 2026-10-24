from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal

#creates langchain  chat-model
grading_model = ChatGroq(model = "openai/gpt-oss-120b", temperature = 0)

class DocumentGrade(BaseModel):
    """Binary relevance score for retrieved documents."""
    binary_score: Literal["yes","no"] = Field(
        description="'yes' if documents are relevant to the question, 'no' otherwise."
    )
 
 #instruction for grading documents   
doc_grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict grader assessing whether retrieved documents are relevant to the user question."
     "Give a binary 'yes' or 'no'. 'yes' means the documents contain information that helps "
     "answer the question, even partially.Say 'yes' ONLY if the documents contain the specific information"
     "needed to answer the question about the specific subject asked."
     "Say 'no' if the documents only mention a similar topic, a different person or entity, "
     "or if the question refers to private or personal information that public documents could not know."),
    ("human","Retrieved documents:\n\n{documents}\n\nQuestion: {question}")
])

doc_grader = doc_grader_prompt|grading_model.with_structured_output(DocumentGrade)

class AnswerGrade(BaseModel):
    """Grades whether an answer is grounded in the documents and addresses the question."""
    grounded: str = Field(description="'yes' if the answer is supported by the documents, 'no' otherwise")
    addresses_question: str = Field(description = "'yes' if the answer resolves the question, 'no' otherwise")
    
answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader. Given documents, a question, and a generated answer, decide:\n"
     "1. Is the answer grounded in (supported by) the documents?\n"
     "2. Does the answer actually address the question?\n"
     "Give binary 'yes'/'no' for each."),
    ("human","Documents:\n\n{documents}\n\nQuestion: {question}\n\nAnswer: {answer}")
])   

answer_grader = answer_grader_prompt | grading_model.with_structured_output(AnswerGrade)

class QuestionCheck(BaseModel):
    answerable_publicly: str = Field(
        description="'no' if the question asks about private or personal information "
                    "that no document or website could know (e.g. 'my neighbour's cat', 'my boss's salary'), else 'yes'"
    )

check_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decide whether a question could be answered from documents or the public web."),
    ("human", "{question}"),
])

question_checker = check_prompt | grading_model.with_structured_output(QuestionCheck)

class RewrittenQuery(BaseModel):
    """A rewritten, standalone search query."""
    rewritten_query: str = Field(description="A clearer, more specific version of the search query")


rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite search queries to be more specific and likely to retrieve relevant documents. "
     "The previous query did not retrieve good results. Make the new query more specific, "
     "expand abbreviations, and consider rephrasing using different keywords."
     "Keep every constraint from the original question (such as 'my', 'names', 'places')."
     "Do not invent dates, numbers, or names. Today's date is {today}."
     "For 'current' or 'today' questions, use the words 'current'  or 'live' instead of a specific date."),
    ("human", "Original question: {question}\nPrevious search query: {search_query}\n\nRewrite it.")
])

query_rewriter = rewrite_prompt | grading_model.with_structured_output(RewrittenQuery)
    