import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()

# --- tools ---
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

# --- agent with human-in-the-loop via interrupts ---
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)

# --- interactive loop ---
def main():
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user = input("\nYou: ").strip()
        if user.lower() == "quit":
            print("👋 Goodbye!")
            break
        for chunk in agent.stream({"messages": [HumanMessage(user)]}, config, stream_mode="values"):
            last = chunk["messages"][-1]
            if last.type == "ai" and last.content:
                print("🤖", last.content)

if __name__ == "__main__":
    main()