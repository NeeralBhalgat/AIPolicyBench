### Documentation Index (Concise)

Start Here
- QUICK_COMMANDS.md: copy/paste commands to run agents, evaluate, reproduce

Core Docs
- ARCHITECTURE.md: high‑level architecture, data flow, key files
- LLM_JUDGE.md: using LLM‑as‑a‑judge (short)
- BENCHMARK_NOTES.md: implementation notes, updates, model config

What You’ll Do Most
- Build vector DB (once):
  - python simple_vector_db.py --json_file data/safety_datasets.json --save
- Run white agent (RAG):
  - python main.py white --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --model deepseek-chat
- Run green agent (evaluator):
  - python main.py green
- One‑shot evaluation:
  - python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat

LLM‑Judge Option
- Add --llm-judge to launch; ensure API keys in .env

Key Files
- RAG: safety_datasets_rag.py
- Vector DB: simple_vector_db.py
- Evaluators: green_agent/evaluation.py
- A2A green: green_agent/a2a_evaluator.py
- A2A white: white_agent/agent.py
- A2A utils: utils/a2a_client.py

Data & Results
- Queries: data/predefined_queries.json
- Vector DB: vector_db/safety_datasets_tfidf_db.pkl
- Results: results/<model>/(summary.json, statistics.txt)

Troubleshooting
- Bad flags: use hyphenated Typer options (e.g., --vector-db)
- DB missing: rebuild via simple_vector_db.py --save
- Agents not ready: check ports 9001/9002
