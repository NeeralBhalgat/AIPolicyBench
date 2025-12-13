# AIPolicyBench

A benchmark for evaluating RAG agents on AI policy questions. Compares Web Search RAG vs Direct LLM approaches using LLM-as-a-judge evaluation.

## Features

- **Web Search RAG**: Real-time web search → scrape → index → retrieve → generate
- **Direct LLM**: Query models directly without retrieval
- **LLM-as-a-Judge**: GPT-4o-mini evaluates responses against ground truth
- **Multi-Agent**: Deploy 6 white agents with different models/modes simultaneously
- **4-Class Evaluation**: Correct ✅ | Miss ⚠️ | Hallucination ❌

## Quick Start

```bash
# Setup
pip install -r requirements.txt
cp env.example .env  # Add OPENROUTER_API_KEY

# Deploy all agents (1 green + 6 white)
./start_multi_agents_cloudflare.sh

# Run evaluation
python send_eval_task.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentBeats Platform                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Green Agent (Evaluator)          White Agents (6x)         │
│   ├── Port 8010                    ├── Ports 8011-8016       │
│   ├── LLM Judge (GPT-4o-mini)      ├── Web RAG (3 agents)    │
│   └── Sends 300 queries            └── Direct LLM (3 agents) │
│              │                              │                │
│              └────── A2A Protocol ──────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### White Agent Modes

| Mode | Pipeline | Use Case |
|------|----------|----------|
| **Web RAG** | Query → Rewrite → Search → Scrape → Index → Retrieve → Generate | Grounded answers with sources |
| **Direct LLM** | Query → Generate | Test model's internal knowledge |

### Evaluation Classes

| Class | Symbol | Description | In Factuality? |
|-------|--------|-------------|----------------|
| Correct | ✅ | Answer matches ground truth | Yes (+) |
| Miss | ⚠️ | Model says "I don't know" | Yes (+) |
| Hallucination | ❌ | Confident but wrong | Yes (-) |

### Metrics

```
Factuality% = Correct% + Miss% − Hallucination%

(Timeout responses are excluded from calculation)
```

## Configuration

### White Agents (start_multi_agents_cloudflare.sh)

| Agent | Port | Model | Mode |
|-------|------|-------|------|
| 1 | 8011 | mistralai/mistral-7b-instruct | Web RAG |
| 2 | 8012 | deepseek/deepseek-v3.2-exp | Web RAG |
| 3 | 8013 | openai/gpt-5.1 | Web RAG |
| 4 | 8014 | mistralai/mistral-7b-instruct | Direct LLM |
| 5 | 8015 | deepseek/deepseek-v3.2-exp | Direct LLM |
| 6 | 8016 | openai/gpt-5.1 | Direct LLM |

### Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...  # Required for LLM calls
```

## Project Structure

```
AIPolicyBench/
├── main.py                          # CLI entry point
├── send_eval_task.py                # Send evaluation to green agent
├── start_multi_agents_cloudflare.sh # Deploy with Cloudflare tunnels
├── white_agents_config.json         # Auto-generated agent URLs
├── src/
│   ├── green_agent/
│   │   ├── a2a_evaluator.py         # Main evaluator logic
│   │   └── evaluation.py            # LLM Judge implementation
│   ├── white_agent/
│   │   ├── agent.py                 # A2A agent server
│   │   ├── pipeline.py              # Web Search RAG pipeline
│   │   └── direct_llm.py            # Direct LLM mode
│   ├── utils/
│   │   ├── llm_client.py            # OpenRouter/OpenAI client
│   │   └── a2a_client.py            # Agent-to-agent communication
│   └── config.py                    # Configuration
├── data/
│   ├── predefined_queries.json      # 300 queries with ground truth
│   └── safety_datasets.json         # Source documents
├── white_agent_{1-6}/               # Separate workspaces per agent
└── results/new_white_agent_design/  # Evaluation outputs
```

## Commands

```bash
# Full evaluation (all 300 queries)
python send_eval_task.py

# Quick test (10 queries)
python send_eval_task.py --max-queries 10

# Custom green agent URL
python send_eval_task.py --green-url http://localhost:8010/to_agent/<id>

# View results
ls results/new_white_agent_design/
cat results/new_white_agent_design/*/statistics.txt
```

## Results Format

Each agent's results in `results/new_white_agent_design/<model>_<mode>/`:

```
statistics.txt          # Summary stats
summary.json            # Machine-readable summary
query_001.json          # Individual query result
query_002.json
...
```

### Example statistics.txt

```
Total Queries: 300
Evaluated (excl. timeout): 285
Timeout: 15 (5.00%) - excluded from factuality

Correct: 85 (29.82%)
Miss: 50 (17.54%)
Hallucination: 150 (52.63%)
Factuality Rate: -5.26%
```

## Performance Optimizations

- **Parallel URL scraping** with aiohttp (5 concurrent)
- **95s internal timeout** to avoid Cloudflare 524 errors
- **Snippet fallback** when scraping fails
- **Domain scoring** prioritizes .gov, .edu sources

## Requirements

- Python 3.11+
- OpenRouter API key (or OpenAI/Anthropic)
- cloudflared (for public deployment)

```bash
pip install -r requirements.txt
```
