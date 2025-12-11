# White Agent #4 Configuration

**Model**: `mistralai/mistral-7b-instruct`
**Mode**: Direct LLM (No RAG)
**Port**: 8014

## Environment Variables

```bash
export AGENT_TYPE=white
export PORT=8014
export WHITE_AGENT_MODEL=mistralai/mistral-7b-instruct
export USE_DIRECT_LLM=true
```

## Run Command

```bash
# From this directory
./run.sh

# Or with agentbeats
agentbeats run_ctrl
```
