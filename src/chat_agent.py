import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# --- tools (add more later) ---
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

tools = [add_numbers]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- one-line agent ---
agent = create_react_agent(llm, tools)

# --- interactive loop ---
def main():
    print("🤖 Agent ready. Type 'quit' to exit.")
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
        messages.append(result["messages"][-1])  # keep history

if __name__ == "__main__":
    main()