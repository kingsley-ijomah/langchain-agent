# LangChain → LangGraph → LangGraph Platform Cheat Sheet

This repository provides a quick setup for working with LangChain, LangGraph, and the LangGraph Platform.

## Quick Setup

### 1. Create Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Required Packages

```bash
pip install langchain langgraph langchain-openai langchain-chroma sentence-transformers python-dotenv fastapi uvicorn langserve langgraph-cli
```

### 3. Set Up Environment Variables

Create a `.env` file with your API keys:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=demo-project
EOF
```

### 4. Create Project Structure

```bash
mkdir -p src
```

## Project Files

The setup creates the following files:

- `src/__init__.py` - Python package initialization
- `src/hello_agent.py` - CLI agent placeholder
- `src/api.py` - FastAPI endpoint placeholder  
- `src/agent_graph.py` - LangGraph Platform entry placeholder
- `langgraph.json` - LangGraph configuration

## Usage

### Run CLI Agent

```bash
python -m src.hello_agent
```

### Start FastAPI Server

```bash
uvicorn src.api:app --reload --port 8000
```

### Test API Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "123 + 456?"}'
```

### Launch LangGraph Platform

```bash
langgraph dev
```

This opens the LangGraph Studio at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024