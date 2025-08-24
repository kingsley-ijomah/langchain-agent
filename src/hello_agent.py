import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

class State(TypedDict):
    question: str
    final_answer: str

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- single node: the LLM decides what to do ---
def decide(state: State):
    prompt = (
        "You are a helpful assistant.\n"
        f"User: {state['question']}\n"
        "Reply with a short answer."
    )
    answer = llm.invoke(prompt).content
    return {"final_answer": answer}

# --- wire 1-node graph ---
g = StateGraph(State)
g.add_node("agent", decide)
g.set_entry_point("agent")
g.add_edge("agent", END)

app = g.compile()

# --- run once ---
if __name__ == "__main__":
    question = "What is 2+2?"
    result = app.invoke({"question": question, "final_answer": ""})
    print("🤖", result["final_answer"])