python -m venv venv

source venv/bin/activate

touch requirements.txt

python -m src.hello_agent

uvicorn src.api:app --reload --port 8000

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "123 + 456?"}'

langgraph dev