<<<<<<< HEAD
# AIPolicyBench — README

## Project Structure
```bash
src/
├── green_agent/    # Benchmark agent (agent.py)
├── white_agent/    # Agent that is completing the task (agent.py)
└── launcher.py     # Evaluation coordinator / CLI wrapper (launches full runs)
```

Other important files
- simple_vector_db.py   — build/load retrieval DB
- data/                 — datasets and predefined queries
- results/              — outputs (white agent responses, evaluations)
- agent-leaderboard-web — React frontend (visualize leaderboard)

## Installation
1. Install Python deps:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp env.example .env
# edit .env and set OPENAI_API_KEY=your_key_here (or other provider keys)
```

3. Project sync step (per infra):
```bash
uv sync
```

## Usage
First, configure .env with OPENAI_API_KEY=...

Launch full evaluation: 
```bash
python main.py launch
```
This command will run white agents to complete the task and green agent to evaluate the white agents all at once.
(Note: `main.py` / `launcher.py` is the coordinator that runs white-agent generation, calls green-agent evaluation, and writes results/leaderboard.json. If missing, run the evaluation CLI directly as shown below.)

## Green Agent's Evaluation
We will be evaluating green agent by randomly sampling a set of green agent outputs and manually check its assessment of hallucination. Since our green agent is a QA policy research agent, we have employed LLM-as-judge for our green agent. Therefore, we cannot evalute our green agent using LLM-as-judge. Therefore, we decided to manually check it.
=======
### AIPolicyBench – README

What is this?
- RAG agent answers grounded AI policy questions by retrieving relevant document chunks and synthesizing concise, source‑aligned responses.
- Solves factual QA over policy PDFs by combining TF‑IDF retrieval with targeted LLM generation to minimize unsupported claims.
- Green agent defines benchmarks using LLM‑as‑a‑judge and/or rule‑based scoring to classify each answer as correct, miss, or hallucination.
- Correct = aligns with ground truth; Hallucination = confident but wrong; Miss = expresses uncertainty (e.g., “I don’t know”).
- Metrics: Correct% = correct/total; Miss% = misses/total; Hallucination% = hallucinations/total; Factuality% = Correct% − Hallucination% + 0.5×Miss%.
- The judge returns classification, confidence, and brief reasoning per query; we also offer fast rule‑based evaluation.
- We evaluate multiple LLM evaluation models; per‑model results are saved under results/<model>/ as summary.json and statistics.txt.
- White agent implements RAG (retrieve → augment → generate) over a prebuilt TF‑IDF vector database for fast, deterministic retrieval.
- It exposes an A2A HTTP endpoint and uses the selected LLM (`--model`) to generate concise, document‑grounded answers.

Other important files
- `simple_vector_db.py` — build/load retrieval DB
- `data/` — datasets and predefined queries
- `results/` — outputs (white agent responses, evaluations)

Setup
```bash
pip install -r requirements.txt
cp env.example .env   # add API keys if using LLMs (e.g., DEEPSEEK_API_KEY or OPENAI_API_KEY)
```

Build the vector DB (once)
```bash
python simple_vector_db.py --json_file data/safety_datasets.json --save
# creates: ./vector_db/safety_datasets_tfidf_db.pkl
```

Run white agent (serve RAG answers)
```bash
python main.py white --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --model deepseek-chat
# options: --host 0.0.0.0 --port 9002
```

Run green agent (evaluate a white agent)
```bash
python main.py green
# options: --host 0.0.0.0 --port 9001
```

One‑shot evaluation (starts green + white and runs benchmark)
```bash
python main.py launch \
  --queries-file data/predefined_queries.json \
  --vector-db ./vector_db/safety_datasets_tfidf_db.pkl \
  --white-model deepseek-chat
```

Use LLM‑as‑a‑judge (optional)
```bash
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl \
  --white-model deepseek-chat --llm-judge
```

Test green agent’s evaluation on sample cases
```bash
# Rule-based demo (no API needed)
python green_agent/evaluation.py

# Predefined queries (rule‑based by default)
python green_agent/agent.py --query_id 1
python green_agent/agent.py --all

# Smoke tests
python test_a2a_imports.py
python test_llm_judge.py
```

Reproduce benchmark variants
```bash
# DeepSeek (default)
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat
# Mistral
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model mistralai/mistral-7b-instruct
# OpenRouter / OpenAI‑compatible
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model openai/gpt-5.1
```

Outputs
- Results in `./results/<model>/`: `summary.json`, `statistics.txt`, per‑query JSON.

AgentBeats compatibility
- Both agents run as A2A HTTP servers (Starlette via a2a‑sdk).
- Start with public bind when needed:
```bash
python main.py green --host 0.0.0.0 --port 9001
python main.py white --host 0.0.0.0 --port 9002 --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --model deepseek-chat
```
- The green agent accepts a standard A2A message containing tags:
  - `<white_agent_url>http://HOST:9002</white_agent_url>`
  - `<queries_file>data/predefined_queries.json</queries_file>`
  - `<use_llm_judge>true|false</use_llm_judge>`

More
- See `misc/QUICK_COMMANDS.md` for a one‑pager of all commands.
>>>>>>> 66d7a65b6abc8b3d7ea63a4821ddc611bc7508f2

