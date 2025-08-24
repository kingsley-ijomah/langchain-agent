from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# build retriever once
facts = [Document(page_content="The capital of France is Paris.")]
vectorstore = Chroma.from_documents(
    facts,
    OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

@tool
def search_docs(query: str) -> str:
    """Look up facts in the local knowledge base."""
    return "\n".join(doc.page_content for doc in retriever.invoke(query))

tools = [add_numbers, search_docs]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

graph = create_react_agent(llm, tools)