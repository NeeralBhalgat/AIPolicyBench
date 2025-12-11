# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AgentBeats Platform                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐           ┌──────────────────────────────┐   │
│   │   Green Agent    │           │       White Agents (6x)       │   │
│   │   (Evaluator)    │           │                               │   │
│   │                  │   A2A     │  ┌─────────┐ ┌─────────┐      │   │
│   │  ┌────────────┐  │ Protocol  │  │ RAG #1  │ │ RAG #2  │ ...  │   │
│   │  │ LLM Judge  │  │ ────────► │  │ Mistral │ │DeepSeek │      │   │
│   │  │(GPT-4o-mini│  │           │  └─────────┘ └─────────┘      │   │
│   │  └────────────┘  │           │  ┌─────────┐ ┌─────────┐      │   │
│   │                  │           │  │Direct #1│ │Direct #2│ ...  │   │
│   │  Port 8010       │           │  │ Mistral │ │DeepSeek │      │   │
│   └──────────────────┘           │  └─────────┘ └─────────┘      │   │
│                                  │  Ports 8011-8016              │   │
│                                  └──────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### Green Agent (Evaluator)
- **Location**: `src/green_agent/`
- **Port**: 8010
- **Function**: Receives evaluation tasks, queries white agents, judges responses
- **Key Files**:
  - `a2a_evaluator.py` - Main A2A server and evaluation logic
  - `evaluation.py` - LLM Judge implementation

### White Agent (RAG/Direct)
- **Location**: `src/white_agent/`
- **Ports**: 8011-8016
- **Function**: Answers AI policy questions
- **Key Files**:
  - `agent.py` - A2A server
  - `pipeline.py` - Web Search RAG pipeline
  - `direct_llm.py` - Direct LLM mode

## Data Flow

### Web RAG Mode
```
Query
  ↓
LLM Rewrite Query (optimize for search)
  ↓
DuckDuckGo Search (10 results)
  ↓
Parallel URL Scraping (aiohttp, 3s timeout)
  ↓
TF-IDF Index Build
  ↓
Top-K Retrieval
  ↓
LLM Generation (with context)
  ↓
Response
```

### Direct LLM Mode
```
Query → LLM Generation → Response
```

### Evaluation Flow
```
Green Agent receives task
  ↓
For each White Agent:
  ├── Send 300 queries (batch of 4)
  ├── Collect responses
  ├── LLM Judge evaluates each
  │   ├── Correct ✅
  │   ├── Miss ⚠️
  │   ├── Hallucination ❌
  │   └── Timeout ⏱️
  └── Save results
  ↓
Calculate Factuality Rate
  ↓
Save summary.json + statistics.txt
```

## Evaluation Classes

| Class | Criteria | Example |
|-------|----------|---------|
| **Correct** | Response contains ground truth | Q: "What are the 3 pillars?" A: "Innovation, infrastructure, diplomacy" |
| **Miss** | Response expresses uncertainty | "I don't have enough information" |
| **Hallucination** | Response is confident but wrong | Q: "Who signed X?" A: "Biden" (when it was Trump) |
| **Timeout** | Response exceeded 95s limit | `[TIMEOUT] Query processing exceeded time limit.` |

## Metrics

```python
# Timeout responses excluded from calculation
evaluated_total = total - timeout_count

correct_rate = correct_count / evaluated_total * 100
miss_rate = miss_count / evaluated_total * 100
hallucination_rate = hallucination_count / evaluated_total * 100

factuality_rate = correct_rate + miss_rate - hallucination_rate
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `send_eval_task.py` | Send evaluation task to green agent |
| `start_multi_agents_cloudflare.sh` | Deploy all agents |
| `white_agents_config.json` | Agent URL mapping (auto-generated) |
| `src/config.py` | Configuration (MAX_SEARCH_RESULTS, etc.) |
| `src/utils/llm_client.py` | OpenRouter/OpenAI client |
| `src/utils/a2a_client.py` | Agent-to-agent communication |
