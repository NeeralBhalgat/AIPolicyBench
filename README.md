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

Frontend / leaderboard
- The frontend visualizes evaluation outputs. Copy an evaluation JSON into agent-leaderboard-web/src/data/ (or point the UI to results/) and run:
```bash
cd agent-leaderboard-web
npm run dev
# Open http://localhost:5173
```
