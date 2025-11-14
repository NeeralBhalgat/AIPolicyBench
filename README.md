# AIPolicyBench — README

## Project Structure
src/
├── green_agent/    # Assessment manager agent (evaluation logic — maps to evaluation.py)
├── white_agent/    # Target agent drivers (generation logic)
└── launcher.py     # Evaluation coordinator / CLI wrapper (launches full runs)

Other important files
- evaluation.py         — main CLI for generate / evaluate / test modes
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

3. (Optional) Frontend:
```bash
cd agent-leaderboard-web
npm install
```

4. Project sync step (per infra):
```bash
uv sync
```

## Usage
First, configure .env with OPENAI_API_KEY=...

Launch full evaluation (generate + evaluate + produce leaderboard JSON)
```bash
uv run python main.py launch
```
(Note: `main.py` / `launcher.py` is the coordinator that runs white-agent generation, calls green-agent evaluation, and writes results/leaderboard.json. If missing, run the evaluation CLI directly as shown below.)

Run white agents (generate responses)
- Generate responses for all predefined queries with a chosen LLM (white agent):
```bash
cd /Users/isabelle/Desktop/AIPolicyBench
python evaluation.py --mode generate --all --llm_provider openai --out results/white_openai_all.json
```
- Generate for a single query:
```bash
python evaluation.py --mode generate --query_id 1 --llm_provider openai --out results/white_openai_q1.json
```

Run the green agent (evaluate white agents)
- Evaluate a saved responses file with the green agent (LLM-as-judge):
```bash
python evaluation.py --mode evaluate --responses results/white_openai_all.json \
  --use_llm_judge --llm_provider openai --out results/eval_openai_all.json
```
- Evaluate using the rule-based evaluator (no API):
```bash
python evaluation.py --mode evaluate --responses results/white_openai_all.json --use_rule_based --out results/eval_rule_based.json
```
- Integrated generate + evaluate:
```bash
python evaluation.py --mode generate_and_evaluate --all --llm_provider openai --use_llm_judge --out results/roundtrip_openai.json
```

Testing the green agent’s evaluation on test cases
- Single test-case (evaluate one predefined query):
```bash
python evaluation.py --mode test --query_id 1 --llm_provider openai --use_llm_judge
```
- Batch test (small set or all queries):
```bash
python evaluation.py --mode test --all --llm_provider openai --use_llm_judge --out results/green_test_openai.json
```
Tips:
- Save evaluation JSON and assert expected labels/confidence in your test harness.
- Example assertion: expect label "CORRECT" for query 1 with confidence >= 0.7.

Frontend / leaderboard
- The frontend visualizes evaluation outputs. Copy an evaluation JSON into agent-leaderboard-web/src/data/ (or point the UI to results/) and run:
```bash
cd agent-leaderboard-web
npm run dev
# Open http://localhost:5173
```

## Recommended quick workflow
1. Ensure `.env` has API keys.
2. (Optional) Build vector DB:
   python simple_vector_db.py --json_file data/safety_datasets.json --save
3. Generate white-agent responses:
   python evaluation.py --mode generate --all --llm_provider <white_agent> --out results/white_<agent>.json
4. Evaluate with green agent:
   python evaluation.py --mode evaluate --responses results/white_<agent>.json --use_llm_judge --llm_provider <green_agent> --out results/eval_<agent>.json
5. Open frontend to inspect leaderboard.
