# White Agent #1 Configuration

**Model**: `mistralai/mistral-7b-instruct`
**Mode**: Web RAG
**Port**: 8011

## Environment Variables

```bash
export AGENT_TYPE=white
export PORT=8011
export WHITE_AGENT_MODEL=mistralai/mistral-7b-instruct
export USE_DIRECT_LLM=false
```

## Run Command

```bash
# From this directory
./run.sh

# Or with agentbeats
agentbeats run_ctrl
```
