# White Agent #6 Configuration

**Model**: `openai/gpt-5.1`
**Mode**: Direct LLM (No RAG)
**Port**: 8016

## Environment Variables

```bash
export AGENT_TYPE=white
export PORT=8016
export WHITE_AGENT_MODEL=openai/gpt-5.1
export USE_DIRECT_LLM=true
```

## Run Command

```bash
# From this directory
./run.sh

# Or with agentbeats
agentbeats run_ctrl
```
