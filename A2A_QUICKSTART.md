# AIPolicyBench A2A Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Key new dependencies for A2A:
- `a2a-sdk[http-server]>=0.3.8` - Agent-to-Agent protocol
- `typer>=0.19.2` - CLI framework
- `httpx>=0.25.0` - HTTP client for A2A communication

2. **Verify installation**:
```bash
python test_a2a_imports.py
```

## Quick Usage

### 1. Launch Complete Evaluation (Easiest)

This starts both agents and runs the full evaluation:

```bash
python main.py launch
```

### 2. View Available Commands

```bash
python main.py --help
```

Output:
```
Commands:
  green    Start the green agent (assessment manager/evaluator).
  white    Start the white agent (RAG system being tested).
  launch   Launch the complete A2A evaluation workflow.
  info     Display information about AIPolicyBench.
  version  Display version information.
```

### 3. Manual Agent Control

**Terminal 1 - Start Green Agent (Evaluator)**:
```bash
python main.py green
```

**Terminal 2 - Start White Agent (RAG System)**:
```bash
python main.py white
```

**Terminal 3 - Send Evaluation Task**:
```python
import asyncio
from utils import a2a_client

async def evaluate():
    response = await a2a_client.send_message(
        "http://localhost:9001",
        """<white_agent_url>http://localhost:9002</white_agent_url>
        <queries_file>data/predefined_queries.json</queries_file>
        <use_llm_judge>false</use_llm_judge>"""
    )
    print(response)

asyncio.run(evaluate())
```

## Configuration Options

### Launch Command Options

```bash
python main.py launch [OPTIONS]

Options:
  --queries-file TEXT      Path to queries JSON [default: data/predefined_queries.json]
  --vector-db TEXT         Path to vector database [default: ./vector_db/safety_datasets_tfidf_db.pkl]
  --llm-judge             Use LLM-as-a-judge evaluation
  --green-host TEXT       Green agent host [default: localhost]
  --green-port INTEGER    Green agent port [default: 9001]
  --white-host TEXT       White agent host [default: localhost]
  --white-port INTEGER    White agent port [default: 9002]
  --help                  Show help message
```

### Examples

**Use LLM judge evaluation**:
```bash
python main.py launch --llm-judge
```

**Custom queries and database**:
```bash
python main.py launch \
  --queries-file data/my_queries.json \
  --vector-db ./vector_db/my_custom_db.pkl
```

**Custom ports**:
```bash
python main.py launch --green-port 8001 --white-port 8002
```

## Architecture Overview

```
┌─────────────────┐
│    Launcher     │
│   (launcher.py) │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────┐
│ Green  │  │ White  │
│ Agent  │──│ Agent  │
│ (9001) │  │ (9002) │
└────────┘  └────────┘
Evaluator    RAG System
```

1. **Launcher** starts both agents
2. **Green Agent** (port 9001):
   - Receives evaluation tasks
   - Sends queries to White Agent
   - Evaluates responses
   - Returns metrics

3. **White Agent** (port 9002):
   - Receives policy questions
   - Performs vector search
   - Generates LLM responses
   - Returns answers

## Understanding the Output

When you run `python main.py launch`, you'll see:

```
🚀 Launching AIPolicyBench A2A Evaluation
================================================================================
📗 Launching green agent (evaluator)...
✅ Green agent is ready at http://localhost:9001

📄 Launching white agent (RAG system)...
✅ White agent is ready at http://localhost:9002

📤 Sending evaluation task to green agent...
⏳ Running evaluation (this may take a while)...

================================================================================
📊 EVALUATION RESULTS
================================================================================
✅ Evaluation Complete!

📊 Results Summary:
- Total Queries: 7
- Correct: 5 (71.43%)
- Miss: 1 (14.29%)
- Hallucination: 1 (14.29%)
- Factuality Rate: 85.71%

Evaluation Method: Rule-based
...

🧹 Cleaning up...
✅ Agents terminated successfully
✅ Evaluation complete!
```

## Evaluation Metrics Explained

- **Correct**: Agent provided the right answer
- **Miss**: Agent expressed uncertainty (safe behavior)
- **Hallucination**: Agent provided wrong information
- **Factuality Rate**: Percentage of non-hallucinated responses (Correct + Miss)

Higher factuality rate = More trustworthy agent

## Troubleshooting

### Issue: Port already in use

**Error**: `Address already in use`

**Solution**: Change ports or kill existing processes
```bash
# Kill process on port 9001
lsof -ti:9001 | xargs kill -9

# Or use different ports
python main.py launch --green-port 8001 --white-port 8002
```

### Issue: Dependencies not installed

**Error**: `ModuleNotFoundError: No module named 'a2a'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Vector database not found

**Error**: `Vector database not found`

**Solution**: Ensure vector DB exists or rebuild it
```bash
# Check if it exists
ls -lh vector_db/safety_datasets_tfidf_db.pkl

# If missing, rebuild (see main documentation)
python build_vector_db.py
```

### Issue: Evaluation timeout

**Error**: Evaluation hangs or times out

**Solution**:
1. Check your `.env` file has valid API keys
2. Increase timeout in launcher.py
3. Check white agent logs for errors

## Next Steps

1. **Read full documentation**: See [A2A_README.md](A2A_README.md)
2. **Customize queries**: Edit `data/predefined_queries.json`
3. **Try LLM judge**: Run with `--llm-judge` flag
4. **Explore codebase**: See [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)

## Comparison: Old vs New Interface

### Old Interface (Still Works)
```bash
python green_agent/agent.py --all
```

### New A2A Interface
```bash
python main.py launch
```

**Benefits of A2A**:
- ✅ Standardized agent communication
- ✅ Distributed evaluation possible
- ✅ Interoperable with other A2A agents
- ✅ Better separation of concerns
- ✅ Easier to extend and scale

Both interfaces work! Use the one that fits your needs.
