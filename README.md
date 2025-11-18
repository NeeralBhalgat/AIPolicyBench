### AIPolicyBench – README (Concise)

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

<<<<<<< HEAD
Other important files
- simple_vector_db.py   — build/load retrieval DB
- data/                 — datasets and predefined queries
- results/              — outputs (white agent responses, evaluations)

## Installation
1. Install Python deps:
=======
Setup
>>>>>>> 6f103c31 (docs: consolidate misc into 5 concise guides; add ARCHITECTURE, LLM_JUDGE, BENCHMARK_NOTES; update QUICK_COMMANDS and index; refresh README usage and AgentBeats notes)
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
<<<<<<< HEAD
python main.py launch --llm-judge
```
This command will run white agents to complete the task and green agent to evaluate the white agents all at once.
=======
python main.py green
# options: --host 0.0.0.0 --port 9001
```
>>>>>>> 6f103c31 (docs: consolidate misc into 5 concise guides; add ARCHITECTURE, LLM_JUDGE, BENCHMARK_NOTES; update QUICK_COMMANDS and index; refresh README usage and AgentBeats notes)

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

