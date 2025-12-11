# White Agent #3 Configuration

**Model**: `openai/gpt-5.1`
**Mode**: Web RAG
**Port**: 8013

## Environment Variables

```bash
export AGENT_TYPE=white
export PORT=8013
export WHITE_AGENT_MODEL=openai/gpt-5.1
export USE_DIRECT_LLM=false
```

## Run Command

```bash
# From this directory
./run.sh

# Or with agentbeats
agentbeats run_ctrl
```
