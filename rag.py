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

class State(TypedDict):
    question: str
    docs: List[Document]
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    answer: str


def retrieve(state):
    q=state["question"]
    return {"docs":retriever.invoke(q)}

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
    context="\n\n".join(d.page_content for d in state["docs"]).strip()
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
g=StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)
g.add_node("refine", refine)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "refine")
g.add_edge("refine","generate")
g.add_edge("generate", END)
app=g.compile()

