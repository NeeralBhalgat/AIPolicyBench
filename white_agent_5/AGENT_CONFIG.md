# White Agent #5 Configuration

**Model**: `deepseek/deepseek-v3.2-exp`
**Mode**: Direct LLM (No RAG)
**Port**: 8015

## Environment Variables

```bash
export AGENT_TYPE=white
export PORT=8015
export WHITE_AGENT_MODEL=deepseek/deepseek-v3.2-exp
export USE_DIRECT_LLM=true
```

## Run Command

```bash
# From this directory
./run.sh

# Or with agentbeats
agentbeats run_ctrl
```
