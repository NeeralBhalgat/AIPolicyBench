### Architecture (Concise)

Purpose
- Evaluate a RAG (white) agent with a coordinating (green) agent via A2A.

Components
- White Agent (RAG): `safety_datasets_rag.py`, `white_agent/agent.py`
- Green Agent (Evaluator): `green_agent/evaluation.py`, `green_agent/a2a_evaluator.py`
- Launcher: `launcher.py` (starts agents, sends task to green)
- Vector DB: `simple_vector_db.py` (TF‑IDF), `vector_db/safety_datasets_tfidf_db.pkl`
- Utils: `utils/llm_client.py` (LLMs), `utils/a2a_client.py` (A2A HTTP)

Flow
1) Build vector DB (once) from `data/safety_datasets.json`
2) White loads DB and answers queries
3) Green sends queries, evaluates responses (rule‑based or LLM‑judge)
4) Save per‑query JSON + `summary.json` + `statistics.txt`

A2A Protocol
- HTTP messages; task text includes tags:
  - `<white_agent_url>...</white_agent_url>`
  - `<queries_file>...</queries_file>`
  - `<use_llm_judge>true|false</use_llm_judge>`

Metrics
- Correct% = correct/total ×100
- Miss% = misses/total ×100
- Hallucination% = hallucinations/total ×100
- Factuality% = Correct% - Hallucination% + 0.5×Miss%

Commands
```bash
# Build DB
python simple_vector_db.py --json_file data/safety_datasets.json --save

# One‑shot evaluation
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat

# With judge
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat --llm-judge
```

Key Files
- RAG: `safety_datasets_rag.py`, `simple_vector_db.py`
- Agents: `green_agent/a2a_evaluator.py`, `white_agent/agent.py`
- Evaluators: `green_agent/evaluation.py`
- Launcher: `launcher.py`

