import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ---------- tools ----------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

facts = [Document(page_content="The capital of France is Paris.")]
vectorstore = Chroma.from_documents(
    facts, OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

@tool
def search_docs(query: str) -> str:
    """Look up facts."""
    return "\n".join(doc.page_content for doc in retriever.invoke(query))

tools = [add_numbers, search_docs]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ---------- agent ----------
agent = create_react_agent(llm, tools)

# ---------- FastAPI ----------
app = FastAPI(title="Mini Agent API")

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(q: Query):
    result = agent.invoke({"messages": [HumanMessage(q.message)]})
    return {"answer": result["messages"][-1].content}

@app.get("/")
def root():
    return {"status": "ready"}