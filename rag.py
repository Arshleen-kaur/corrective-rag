from typing import List, TypedDict
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
    model="poolside/laguna-m.1:free"
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)