# White Agent #2 Configuration

**Model**: `deepseek/deepseek-v3.2-exp`
**Mode**: Web RAG
**Port**: 8012

## Environment Variables

```bash
export AGENT_TYPE=white
export PORT=8012
export WHITE_AGENT_MODEL=deepseek/deepseek-v3.2-exp
export USE_DIRECT_LLM=false
```

## Run Command

```bash
# From this directory
./run.sh

# Or with agentbeats
agentbeats run_ctrl
```
