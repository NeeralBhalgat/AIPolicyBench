# White Agent Model Configuration

## Overview

The AIPolicyBench A2A evaluation system now supports **configurable models for the white agent** while keeping the LLM-as-a-judge model fixed at `gpt-4o-mini`.

## Key Changes

### 1. Results Organization

**Before:**
```
./results/
└── localhost_9002/
    └── 20250113_211500/  # Timestamp-based
        ├── query_001.json
        └── summary.json
```

**After:**
```
./results/
└── deepseek-chat/  # Model name-based (no timestamp in path)
    ├── query_001.json
    └── summary.json
```

### 2. White Agent Model Configuration

The white agent can now be configured to use different LLM models:

- **deepseek-chat** (default) - Direct DeepSeek API
- **mistralai/mistral-7b-instruct** - Via OpenRouter
- **openai/gpt-4o-mini** - Via OpenRouter
- **openai/gpt-3.5-turbo** - Via OpenRouter
- Any other model supported by your API provider

### 3. LLM Judge Model (Fixed)

The green agent's LLM-as-a-judge evaluation **always uses gpt-4o-mini**:
- Model: `openai/gpt-4o-mini` via OpenRouter
- Temperature: 0.0
- Max tokens: 800
- This is **not configurable** to ensure consistent evaluation across all white agents

## Usage

### Command Line (Typer CLI)

#### Launch with Default Model (deepseek-chat)
```bash
conda run -n aipolicy python main.py launch --llm-judge
```

#### Launch with Mistral-7B
```bash
conda run -n aipolicy python main.py launch \
  --white-model "mistralai/mistral-7b-instruct" \
  --llm-judge
```

#### Launch with GPT-4o-mini for White Agent
```bash
conda run -n aipolicy python main.py launch \
  --white-model "openai/gpt-4o-mini" \
  --llm-judge
```

#### Launch with GPT-3.5-turbo
```bash
conda run -n aipolicy python main.py launch \
  --white-model "openai/gpt-3.5-turbo" \
  --llm-judge
```

### Start White Agent Separately

```bash
# With deepseek-chat (default)
conda run -n aipolicy python main.py white

# With mistral
conda run -n aipolicy python main.py white \
  --model "mistralai/mistral-7b-instruct"

# With gpt-4o-mini
conda run -n aipolicy python main.py white \
  --model "openai/gpt-4o-mini"
```

### Python API (Launcher)

```python
import asyncio
from launcher import launch_evaluation

# Evaluate with deepseek-chat (default)
asyncio.run(launch_evaluation(
    queries_file="data/predefined_queries.json",
    white_model="deepseek-chat",
    use_llm_judge=True
))

# Evaluate with mistral
asyncio.run(launch_evaluation(
    queries_file="data/predefined_queries.json",
    white_model="mistralai/mistral-7b-instruct",
    use_llm_judge=True
))

# Evaluate with gpt-4o-mini
asyncio.run(launch_evaluation(
    queries_file="data/predefined_queries.json",
    white_model="openai/gpt-4o-mini",
    use_llm_judge=True
))
```

### Direct Python (White Agent)

```python
from white_agent.agent import start_white_agent

# Start with deepseek-chat
start_white_agent(
    vector_db_path="./vector_db/safety_datasets_tfidf_db.pkl",
    model="deepseek-chat",
    host="localhost",
    port=9002
)

# Start with mistral
start_white_agent(
    vector_db_path="./vector_db/safety_datasets_tfidf_db.pkl",
    model="mistralai/mistral-7b-instruct",
    host="localhost",
    port=9002
)
```

## Results Structure

### Summary JSON (`./results/<model>/summary.json`)

```json
{
  "results": [...],
  "statistics": {
    "total": 7,
    "correct": 5,
    "miss": 1,
    "hallucination": 1,
    "correct_rate": 71.43,
    "miss_rate": 14.29,
    "hallucination_rate": 14.29,
    "factuality_rate": 85.71
  },
  "method": "LLM-as-a-judge",
  "white_agent_url": "http://localhost:9002",
  "white_agent_model": "deepseek-chat",
  "llm_judge_provider": "openai/gpt-4o-mini",
  "llm_judge_model": "openai/gpt-4o-mini",
  "timestamp": "20250113_211500",
  "results_dir": "./results/deepseek-chat"
}
```

### Statistics TXT (`./results/<model>/statistics.txt`)

```
AIPolicyBench Evaluation Results
================================================================================

White Agent: http://localhost:9002
White Agent Model: deepseek-chat
Evaluation Method: LLM-as-a-judge
LLM Judge Model: openai/gpt-4o-mini
Timestamp: 20250113_211500

================================================================================
Statistics:
================================================================================

Total Queries: 7
Correct: 5 (71.43%)
Miss: 1 (14.29%)
Hallucination: 1 (14.29%)
Factuality Rate: 85.71%

================================================================================
```

## Model Comparison Workflow

You can now easily compare different models by running evaluations with each:

```bash
# Evaluate deepseek-chat
conda run -n aipolicy python main.py launch \
  --white-model "deepseek-chat" \
  --llm-judge

# Evaluate mistral-7b
conda run -n aipolicy python main.py launch \
  --white-model "mistralai/mistral-7b-instruct" \
  --llm-judge

# Evaluate gpt-4o-mini
conda run -n aipolicy python main.py launch \
  --white-model "openai/gpt-4o-mini" \
  --llm-judge

# Compare results
cat ./results/deepseek-chat/statistics.txt
cat ./results/mistralai_mistral-7b-instruct/statistics.txt
cat ./results/openai_gpt-4o-mini/statistics.txt
```

**Note:** Model names with special characters (/, :, etc.) are sanitized for folder names by replacing them with underscores.

## Files Modified

### 1. [white_agent/agent.py](white_agent/agent.py)

**Added model parameter:**
- `prepare_white_agent_card()` now accepts `model` parameter
- Model name is included in agent card description and tags
- `AIPolityRAGAgentExecutor.__init__()` accepts `model` parameter
- Passes model to `SafetyDatasetsRAG`
- `start_white_agent()` accepts and uses `model` parameter

**Key additions:**
```python
def prepare_white_agent_card(url: str, model: str = "deepseek-chat") -> AgentCard:
    skill = AgentSkill(
        id="ai_policy_rag",
        name="AI Policy RAG",
        description=f"Answers questions about AI safety and policy using RAG over safety datasets (Model: {model})",
        tags=["rag", "ai-safety", "policy", f"model:{model}"],
        # ...
    )
```

### 2. [green_agent/a2a_evaluator.py](green_agent/a2a_evaluator.py)

**Extracts model name from white agent:**
- Fetches white agent card via A2A protocol
- Extracts model name from skill tags (format: `model:deepseek-chat`)
- Uses model name for results folder (no timestamp in path)
- Adds `white_agent_model` to result metadata
- Renames `provider` and `model` to `llm_judge_provider` and `llm_judge_model`

**Key additions:**
```python
# Get white agent card to extract model name
white_agent_card = await a2a_client.get_agent_card(white_agent_url)
if white_agent_card and white_agent_card.skills:
    for skill in white_agent_card.skills:
        for tag in skill.tags:
            if tag.startswith("model:"):
                white_agent_model = tag.replace("model:", "")
                break

# Create results directory based on model name (no timestamp in path)
eval_session_dir = Path(results_dir) / white_agent_model
```

### 3. [launcher.py](launcher.py)

**Added white_model parameter:**
- `launch_evaluation()` accepts `white_model` parameter
- Passes model to `start_white_agent()`
- Command-line argument `--white_model`

**Key additions:**
```python
async def launch_evaluation(
    queries_file: str = "data/predefined_queries.json",
    vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl",
    white_model: str = "deepseek-chat",  # NEW
    use_llm_judge: bool = False,
    # ...
):
    p_white = multiprocessing.Process(
        target=start_white_agent,
        args=(vector_db_path, white_model, white_host, white_port)  # Added white_model
    )
```

### 4. [main.py](main.py)

**Added --white-model option:**
- `white` command accepts `--model` parameter
- `launch` command accepts `--white-model` parameter
- Help text clarifies LLM judge model is fixed at gpt-4o-mini

**Key additions:**
```python
@app.command()
def launch(
    white_model: str = typer.Option(
        "deepseek-chat",
        help="LLM model for white agent (e.g., deepseek-chat, mistralai/mistral-7b-instruct, openai/gpt-4o-mini)"
    ),
    use_llm_judge: bool = typer.Option(
        False,
        "--llm-judge",
        help="Use LLM-as-a-judge evaluation (fixed: gpt-4o-mini)"
    ),
    # ...
):
```

## API Key Configuration

### For White Agent Models

Add to `.env`:

```bash
# For deepseek-chat (direct DeepSeek API)
DEEPSEEK_API_KEY=sk-your-deepseek-key

# For OpenRouter models (mistral, gpt-4o-mini, etc.)
DEEPSEEK_API_KEY=sk-or-your-openrouter-key
```

**Note:** The `SafetyDatasetsRAG` class checks if the API key starts with `sk-or-` to determine whether to use OpenRouter or direct DeepSeek API.

### For LLM Judge (Fixed)

The LLM judge uses OpenRouter with gpt-4o-mini:

```bash
# OpenRouter API key (for LLM judge)
DEEPSEEK_API_KEY=sk-or-your-openrouter-key
```

**Important:** If using different API keys for white agent and LLM judge, you'll need to set them separately in your environment or code.

## Example: Comparing Models

```bash
#!/bin/bash

echo "Evaluating different models..."

# DeepSeek
echo "Testing deepseek-chat..."
conda run -n aipolicy python main.py launch \
  --white-model "deepseek-chat" \
  --llm-judge

# Mistral-7B
echo "Testing mistral-7b..."
conda run -n aipolicy python main.py launch \
  --white-model "mistralai/mistral-7b-instruct" \
  --llm-judge

# GPT-4o-mini
echo "Testing gpt-4o-mini..."
conda run -n aipolicy python main.py launch \
  --white-model "openai/gpt-4o-mini" \
  --llm-judge

# Compare results
echo "\n=== COMPARISON ==="
echo "\nDeepSeek Results:"
cat ./results/deepseek-chat/statistics.txt

echo "\nMistral Results:"
cat ./results/mistralai_mistral-7b-instruct/statistics.txt

echo "\nGPT-4o-mini Results:"
cat ./results/openai_gpt-4o-mini/statistics.txt
```

## Benefits

1. **Organized Results**: Each model's results are stored in a separate folder by model name
2. **No Timestamp Clutter**: Folder structure is cleaner without timestamps in the path
3. **Easy Comparison**: Can directly compare performance across different models
4. **Flexible Testing**: Test any model supported by your API provider
5. **Consistent Evaluation**: LLM judge remains fixed (gpt-4o-mini) for fair comparison
6. **Metadata Preserved**: Timestamp and other metadata still saved in summary files

## Troubleshooting

### Issue: Results folder named "unknown"

**Cause:** Could not extract model name from white agent card

**Solution:**
- Ensure white agent is started with the model parameter
- Check that the agent card includes the model tag
- Verify network connectivity between green and white agents

### Issue: Model not found error

**Cause:** Invalid model name or API provider doesn't support the model

**Solution:**
- Check model name spelling
- Verify API key is for the correct provider
- For OpenRouter models, ensure API key starts with `sk-or-`
- Check model availability at https://openrouter.ai/models

### Issue: Different results for same model

**Cause:** Results overwriting each other in the same folder

**Solution:**
- This is intentional - each model has one results folder
- Results are overwritten on each new evaluation
- If you want to preserve history, manually copy results before re-running:
  ```bash
  cp -r ./results/deepseek-chat ./results/deepseek-chat_backup_$(date +%Y%m%d_%H%M%S)
  ```

## Summary

✅ **White Agent Model**: Configurable via `--white-model` parameter
✅ **LLM Judge Model**: Fixed at `gpt-4o-mini` for consistent evaluation
✅ **Results Organization**: Organized by model name (no timestamp in path)
✅ **Backward Compatible**: Defaults to `deepseek-chat` if not specified
✅ **Full Metadata**: Model names, timestamps, and all info preserved in summary files
✅ **Easy Comparison**: Direct comparison of model performance

The system is now ready for comprehensive model benchmarking! 🎉
