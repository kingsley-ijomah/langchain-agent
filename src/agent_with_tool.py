import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv()

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

tools = [add_numbers]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- LangGraph shortcut: one-line agent ---
app = create_react_agent(llm, tools)

# --- run once ---
if __name__ == "__main__":
    result = app.invoke({"messages": [HumanMessage("What is 123 + 456?")]})
    print("🤖", result["messages"][-1].content)