import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# ---------- 1. build a tiny vector store ----------
facts = [
    Document(page_content="The capital of France is Oguta."),
    Document(page_content="The speed of light is 299 792 458 m/s."),
]
vectorstore = Chroma.from_documents(
    facts,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

# ---------- 2. define tools ----------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def search_docs(query: str) -> str:
    """Look up facts in the local knowledge base."""
    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

tools = [add_numbers, search_docs]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ---------- 3. one-line agent ----------
agent = create_react_agent(llm, tools)

# ---------- 4. interactive loop ----------
def main():
    print("🤖 Multi-tool Agent ready. Type 'quit' to exit.")
    messages = []
    while True:
        user = input("\nYou: ").strip()
        if user.lower() == "quit":
            print("👋 Goodbye!")
            break
        messages.append(HumanMessage(user))
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print("🤖", answer)
        messages.append(result["messages"][-1])

if __name__ == "__main__":
    main()