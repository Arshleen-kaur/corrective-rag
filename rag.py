import re
from typing import List, TypedDict,BaseModel
import time
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

docs =(
    PyPDFLoader("Deep+Learning+Ian+Goodfellow.pdf").load() +
    PyPDFLoader("ml_ebook.pdf").load() +
    PyPDFLoader("Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf").load() 
)
print(len(docs))

# Split the documents into chunks
chunks=RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
#clean the chunks
for d in chunks:
    d.page_content=d.page_content.encode("utf-8", "ignore").decode("utf-8","ignore")
print(len(chunks))

# Index fresh collection each run
embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
vector_store= FAISS.from_documents(chunks,embeddings)
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="poolside/laguna-m.1:free",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

UPPER_TH =0.7
LOWER_TH =0.3

class State(TypedDict):
    question: str
    docs: List[Document]

    good_docs: List[Document]
    verdict: str
    reason:str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    answer: str


def retrieve(state):
    q=state["question"]
    return {"docs":retriever.invoke(q)}

class DocEvalScore(BaseModel):
    score:float
    reason: str
doc_eval_prompt= ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a strict retrieval evaluator for RAG.\n"
         "You will be given one retrieval chunk and a question.\n"
         "Return relevance score in [0.0,1.0]\n"
         "1.0: chunk is fully sufficient to answer the question\n"
         "0.0 chunk is irrelevant\n"
         "Be conservative with high scores.\n"
         "Also return a short reason\n"
         "output json only"),
         ("human","question: {question}\n\n chunk:{chunk}")
    ]
)
doc_eval_chain=doc_eval_prompt|llm.with_structured_output(DocEvalScore)

def eval_each_doc_node(state:State)->State:

    q=state["question"]

    scores:List[float]
    reasons:List[str]
    good:List[Document]

    for d in state["docs"]:
        out=doc_eval_chain.invoke({"question":q,"chunk":d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)

        if out.score>LOWER_TH:
            good.append(d)

    if any(s>UPPER_TH for s in scores):
        return {
            "good_docs":good,
            "verdict":"CORRECT",
            "reason" : f"At least one retrieved chunk stored > {UPPER_TH}.{why}",

        }
    if len(scores)>0 and all(s<LOWER_TH for s in scores):
        why ="No chunk was sufficeient"
        return {
            "good_docs":[],
            "verdict":"INCORRECT",
            "reason":f"All retrieved chunks scored <{LOWER_TH}.{why}",

        }
    why="Mixed relevance signals"
    return {
        "good_docs":good,
        "verdict":"AMBIGUOUS",
        "reason": F"No chunks scored >{UPPER_TH }, but not all were <{LOWER_TH}.{why}",
    }

def decompose_to_sentences(text:str)->List[str]:
    text=re.sub(r"\s+", " ", text).strip()
    sentences=re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip())>20]

class KeepOrDrop(BaseModel):
    keep:bool
filter_prompt =ChatPromptTemplate.from_messages(
    [
        ("system", "You are a strict relevance filter. \nYou will be given a question and a sentence. You will decide if the sentence is relevant to the question. \n\nAnswer only with 'true' or 'false'."),
        ("human","Question:{question}\n\nSentence:{sentence}\n\nAnswer:"),
    ]
)
filter_chain=filter_prompt | llm.with_structured_output(KeepOrDrop)

def refine(state:State)->State:
    q=state["question"]
    context="\n\n".join(d.page_content for d in state["good_docs"]).strip()
    strips=decompose_to_sentences(context)
    kept:List[str]=[]
    for s in strips:
        if filter_chain.invoke({"question":q,"sentence":s}).keep:
            kept.append(s)
    refined_context="\n".join(kept).strip()
    return {
        "strips":strips,
        "kept_strips":kept,
        "refined_context":refined_context,
    }

prompt =ChatPromptTemplate.from_messages(
    [
        ("system", "Answer only from the context. if not in the context say 'I don't know'"),
        ("human","Question:{question}\n\n Context:\n{context}\n\nAnswer:"),
    ]
)
def generate(state):
    context="\,\n".join([d.page_content for d in state["docs"]])
    out=(prompt|llm).invoke({"question":state["question"],"context":context})
    return {"answer":out["text"]}

def fail_node(state:State)->State:
    return {"answer": f"FAIL : {state['reason']}"}
def ambiguous_node(state:State)->State:
    return {"answer":f"Ambiguous : {state['reason']}"}
def router_after_eval(state:State)->str:
    if state["verdict"]=="CORRECT":
        return "refine"
    elif state["verdict"]=="INCORRECT":
        return "fail_node"
    else:
        return "ambiguous"
g=StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("eval_each_doc",eval_each_doc_node)
g.add_node("generate", generate)
g.add_node("refine", refine)
g.add_node("fail",fail_node)
g.add_node("ambiguous",ambiguous_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges(
    "eval_each_doc",
    router_after_eval,
    {"refine":"refine", "web_search":"fail","ambiguous":"ambiguous"}

)
g.add_edge("refine","generate")
g.add_edge("generate", END)
g.add_edge("fail",END)
app=g.compile()

