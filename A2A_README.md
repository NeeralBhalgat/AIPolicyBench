# AIPolicyBench A2A Implementation

This document describes the Agent-to-Agent (A2A) implementation for AIPolicyBench.

## Overview

AIPolicyBench now supports the **A2A (Agent-to-Agent) protocol**, enabling standardized communication between agents and interoperability with other A2A-compliant systems.

## Architecture

### Green Agent (Evaluator)
- **Location**: `green_agent/a2a_evaluator.py`
- **Role**: Assessment manager that orchestrates evaluation
- **Port**: 9001 (default)
- **Functions**:
  - Receives evaluation tasks
  - Loads predefined queries with ground truth
  - Sends queries to white agent
  - Evaluates responses using rule-based or LLM-judge methods
  - Returns detailed evaluation metrics

### White Agent (RAG System)
- **Location**: `white_agent/agent.py`
- **Role**: Target agent being tested (RAG system)
- **Port**: 9002 (default)
- **Functions**:
  - Receives policy questions
  - Retrieves relevant datasets using vector search
  - Generates responses using LLM
  - Returns answers via A2A protocol

### Launcher
- **Location**: `launcher.py`
- **Role**: Coordinates the complete evaluation workflow
- **Functions**:
  - Starts both green and white agents
  - Sends evaluation tasks to green agent
  - Manages agent lifecycle
  - Reports results

## Usage

### Command-Line Interface

The main CLI is in `main.py` and provides several commands:

```bash
# Display information
python main.py info

# Start green agent only (evaluator)
python main.py green

# Start white agent only (RAG system)
python main.py white

# Launch complete evaluation (starts both agents)
python main.py launch

# Use LLM-as-a-judge evaluation
python main.py launch --llm-judge

# Custom configuration
python main.py launch --queries_file data/my_queries.json \
                      --vector_db ./vector_db/my_db.pkl \
                      --green_port 9001 \
                      --white_port 9002
```

### Python API

```python
import asyncio
from launcher import launch_evaluation

# Run evaluation programmatically
asyncio.run(launch_evaluation(
    queries_file="data/predefined_queries.json",
    vector_db_path="./vector_db/safety_datasets_tfidf_db.pkl",
    use_llm_judge=False
))
```

### Manual Agent Control

You can also run agents independently:

```python
# Terminal 1: Start green agent
from green_agent.a2a_evaluator import start_green_agent
start_green_agent(host="localhost", port=9001)

# Terminal 2: Start white agent
from white_agent.agent import start_white_agent
start_white_agent(host="localhost", port=9002)

# Terminal 3: Send evaluation task
import asyncio
from utils import a2a_client

async def send_task():
    response = await a2a_client.send_message(
        "http://localhost:9001",
        """<white_agent_url>http://localhost:9002</white_agent_url>
        <queries_file>data/predefined_queries.json</queries_file>
        <use_llm_judge>false</use_llm_judge>"""
    )
    print(response)

asyncio.run(send_task())
```

## A2A Protocol

### Agent Cards

Each agent exposes an agent card at `/.well-known/agent.json`:

**Green Agent Card** (`green_agent/green_agent.toml`):
- Skill: RAG Agent Assessment
- Evaluates agents using predefined queries
- Returns evaluation metrics

**White Agent Card** (`white_agent/white_agent.toml`):
- Skill: AI Policy RAG
- Answers questions about AI safety datasets
- Uses vector search + LLM generation

### Message Format

Agents communicate using A2A message protocol:

```python
{
    "role": "user",
    "parts": [{"text": "query text"}],
    "message_id": "uuid",
    "task_id": "optional",
    "context_id": "optional"
}
```

### Evaluation Task Format

Green agent expects tasks in this format:

```xml
<white_agent_url>http://localhost:9002</white_agent_url>
<queries_file>data/predefined_queries.json</queries_file>
<use_llm_judge>false</use_llm_judge>
```

## Evaluation Metrics

The green agent returns:

- **Total Queries**: Number of queries evaluated
- **Correct**: Queries answered correctly
- **Miss**: Queries where agent expressed uncertainty
- **Hallucination**: Queries with incorrect answers
- **Correct Rate**: Percentage of correct answers
- **Miss Rate**: Percentage of uncertain responses
- **Hallucination Rate**: Percentage of incorrect answers
- **Factuality Rate**: Correct + Miss (not hallucinating)

## File Structure

```
AIPolicyBench/
├── main.py                          # CLI entry point
├── launcher.py                      # Evaluation coordinator
├── green_agent/
│   ├── agent.py                     # Original interface (backward compatible)
│   ├── a2a_evaluator.py            # A2A evaluator agent
│   ├── green_agent.toml            # Agent card configuration
│   └── evaluation.py                # Evaluation logic
├── white_agent/
│   ├── __init__.py
│   ├── agent.py                     # A2A RAG agent
│   └── white_agent.toml            # Agent card configuration
└── utils/
    ├── a2a_client.py               # A2A client utilities
    ├── parsing.py                   # Tag parsing utilities
    └── llm_client.py               # LLM client (moved from white_agent)
```

## Dependencies

Key A2A-related dependencies:

```
a2a-sdk[http-server]>=0.3.8
typer>=0.19.2
httpx>=0.25.0
uvicorn>=0.37.0
starlette>=0.35.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Backward Compatibility

The original interface in `green_agent/agent.py` (PredefinedQueryInterface) remains unchanged and can still be used:

```bash
# Original usage (still works)
python green_agent/agent.py --all
python green_agent/agent.py --query_id 1
```

## Comparison with Tau-Bench

This implementation is inspired by [tau-bench's agentify example](https://github.com/sierra-research/tau-bench):

| Feature | Tau-Bench | AIPolicyBench |
|---------|-----------|---------------|
| Green Agent | Tool-calling assessment | RAG evaluation |
| White Agent | General LLM agent | Specialized RAG system |
| Evaluation | Task success/failure | Correctness, miss, hallucination |
| Domain | Retail/Airline customer service | AI safety & policy |
| Task Format | Tool calling + user simulation | Question answering |

## Future Enhancements

- [ ] Support for streaming responses
- [ ] Multi-task evaluation (batch processing)
- [ ] Advanced LLM-as-a-judge implementation
- [ ] Integration with other A2A agents
- [ ] Distributed evaluation across multiple agents
- [ ] Real-time evaluation dashboard

## Troubleshooting

### Agents not starting
- Check if ports 9001 and 9002 are available
- Verify vector database exists at the specified path
- Ensure all dependencies are installed

### Evaluation timeout
- Increase timeout in launcher: `timeout=300.0` (5 minutes)
- Check white agent logs for slow LLM responses
- Verify API keys are correctly set in `.env`

### Import errors
```bash
pip install --upgrade a2a-sdk httpx typer uvicorn
```

## References

- [A2A Protocol Specification](https://github.com/anthropics/a2a)
- [Tau-Bench](https://github.com/sierra-research/tau-bench)
- Original AIPolicyBench documentation in `DOCUMENTATION_INDEX.md`
