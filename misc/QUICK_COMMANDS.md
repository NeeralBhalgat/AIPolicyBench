# Quick Commands

## Deploy & Evaluate

```bash
# Start all agents with Cloudflare tunnels
./start_multi_agents_cloudflare.sh

# Run full evaluation (300 queries × 6 agents)
python send_eval_task.py

# Quick test (10 queries)
python send_eval_task.py --max-queries 10
```

## View Results

```bash
# List all results
ls results/new_white_agent_design/

# View statistics for all agents
cat results/new_white_agent_design/*/statistics.txt

# View specific agent
cat results/new_white_agent_design/openai-gpt-5-1_rag/statistics.txt
```

## Manual Agent Control

```bash
# Green agent (evaluator)
python main.py green --host 0.0.0.0 --port 9001

# White agent (RAG)
python main.py white --host 0.0.0.0 --port 9002 --model deepseek-chat

# White agent (Direct LLM)
python main.py white --host 0.0.0.0 --port 9002 --model deepseek-chat --direct-llm
```

## Logs

```bash
# Green agent
tail -f /tmp/green_agent.log

# White agents
tail -f /tmp/white_agent_8011.log  # Mistral RAG
tail -f /tmp/white_agent_8012.log  # DeepSeek RAG
tail -f /tmp/white_agent_8013.log  # GPT-5.1 RAG
tail -f /tmp/white_agent_8014.log  # Mistral Direct
tail -f /tmp/white_agent_8015.log  # DeepSeek Direct
tail -f /tmp/white_agent_8016.log  # GPT-5.1 Direct
```

## Troubleshooting

```bash
# Check agent URLs
cat white_agents_config.json

# Test white agent
curl https://<cloudflare-url>/to_agent/<id>/.well-known/agent-card.json

# Check for errors
grep -i error /tmp/white_agent_8011.log | tail -10
```
