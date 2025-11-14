# GPT-4o-mini LLM Judge with Enhanced Logging & Results Saving

## Overview

Updated the LLM-as-a-judge evaluation to use **GPT-4o-mini** via OpenRouter with comprehensive logging, progress tracking, and automatic results saving.

## Key Changes

### 1. Model Updated: GPT-4o-mini ✅
- **Previous**: `mistralai/mistral-7b-instruct`
- **New**: `openai/gpt-4o-mini`
- **Benefits**:
  - Higher quality judgments
  - Better JSON formatting
  - More reliable reasoning
  - Good cost-performance ratio

### 2. Progress Tracking with tqdm ✅
- Real-time progress bar showing:
  - Current query number
  - Total progress percentage
  - Live statistics (Correct, Miss, Hallucination)
- Example output:
  ```
  Query 3/7: 43%|████▎     | 3/7 [00:12<00:16, 4.1s/query] Correct: 2, Miss: 0, Halluc: 1
  ```

### 3. Detailed Logging for Each Evaluation ✅
Enhanced logging includes:
- **Query Information**: ID, question, ground truth
- **White Agent Response**: Full response text
- **Evaluation Method**: LLM-judge or Rule-based
- **Judgment**: Result with emoji indicators (✅ ❌ ⚠️)
- **Confidence Score**: 0.0 to 1.0 (LLM-judge only)
- **Reasoning**: Detailed explanation (LLM-judge only)
- **File Saved**: Path to individual result file

#### Example Log Output:
```
================================================================================
[Query 1] Are there datasets available for AI safety research?
[Ground Truth] yes
================================================================================
[White Agent Response]
Yes, there are several datasets available for AI safety research...

[Evaluating with LLM-as-a-judge]

[Judgment] ✅ CORRECT
[Confidence] 0.95
[Reasoning] The response correctly states that there are datasets available...
[Saved] ./results/localhost_9002/20250113_211500/query_001.json
```

### 4. Automatic Results Saving ✅

#### Directory Structure:
```
./results/
└── <white_agent_name>/        # e.g., localhost_9002
    └── <timestamp>/            # e.g., 20250113_211500
        ├── query_001.json      # Individual query result
        ├── query_002.json
        ├── query_003.json
        ├── ...
        ├── summary.json        # Complete evaluation summary
        └── statistics.txt      # Human-readable statistics
```

#### Individual Query File (query_001.json):
```json
{
  "query_id": 1,
  "query": "Are there datasets available for AI safety research?",
  "response": "Yes, there are several datasets...",
  "ground_truth": "yes",
  "evaluation_result": "correct",
  "evaluation_method": "LLM-as-a-judge",
  "timestamp": "2025-01-13T21:15:23.456789",
  "confidence": 0.95,
  "reasoning": "The response correctly states...",
  "provider": "deepseek"
}
```

#### Summary File (summary.json):
```json
{
  "results": [...],  // All individual results
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
  "provider": "openai/gpt-4o-mini",
  "model": "openai/gpt-4o-mini",
  "white_agent_url": "http://localhost:9002",
  "queries_file": "data/predefined_queries.json",
  "timestamp": "20250113_211500",
  "results_dir": "./results/localhost_9002/20250113_211500"
}
```

#### Statistics File (statistics.txt):
```
AIPolicyBench Evaluation Results
================================================================================

White Agent: http://localhost:9002
Evaluation Method: LLM-as-a-judge
Model: openai/gpt-4o-mini
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

## Files Modified

### 1. [green_agent/a2a_evaluator.py](green_agent/a2a_evaluator.py)

**Imports Added**:
```python
import os
from datetime import datetime
from tqdm import tqdm
```

**Function Signature Updated**:
```python
async def evaluate_white_agent(
    white_agent_url: str,
    queries_file: str = "data/predefined_queries.json",
    use_llm_judge: bool = False,
    results_dir: str = "./results"  # NEW
) -> dict:
```

**Key Changes**:
- Create results directory structure based on white agent name and timestamp
- Initialize evaluator with GPT-4o-mini (line 110-116)
- Add tqdm progress bar (line 133)
- Enhanced logging for each query (lines 143-146, 187-188, 211-215)
- Save individual query results to JSON files (lines 246-250)
- Update progress bar with live statistics (lines 253-257)
- Close progress bar after completion (line 270)
- Enhanced final logging (lines 279-285)
- Save summary.json and statistics.txt (lines 312-337)
- Update provider info to gpt-4o-mini (lines 308-310)

### 2. [test_llm_judge.py](test_llm_judge.py)

**Updated Model**:
```python
llm_evaluator = LLMJudgeEvaluator(
    provider="deepseek",
    model="openai/gpt-4o-mini",  # Changed from mistral
    temperature=0.0,
    max_tokens=800  # Increased from 500
)
```

## Usage

### Command Line:
```bash
# Launch with LLM judge (will save results automatically)
conda run -n aipolicy python main.py launch --llm-judge
```

### Check Results:
```bash
# View latest evaluation results
ls -lt ./results/localhost_9002/

# Read statistics
cat ./results/localhost_9002/<timestamp>/statistics.txt

# View individual query result
cat ./results/localhost_9002/<timestamp>/query_001.json
```

### Monitor Progress:
When running evaluation, you'll see:
```
================================================================================
Starting evaluation of 7 queries
================================================================================

Query 1/7: 14%|█▍        | 1/7 [00:04<00:24, 4.1s/query] Correct: 1, Miss: 0, Halluc: 0
```

## Benefits

### 1. Better Model Quality
- **GPT-4o-mini** provides more accurate judgments than Mistral-7B
- Better understanding of semantic meaning
- More consistent JSON formatting
- Higher confidence scores

### 2. Complete Traceability
- Every query result saved individually
- Timestamped evaluation sessions
- Full audit trail of all judgments
- Easy to review and analyze later

### 3. Real-Time Monitoring
- Progress bar shows live progress
- See statistics update in real-time
- Estimate time remaining
- Track success rate during evaluation

### 4. Easy Analysis
- Results organized by white agent
- Multiple evaluation sessions preserved
- JSON files for programmatic analysis
- Human-readable statistics file

## Configuration

### Change Results Directory:
```python
result = await evaluate_white_agent(
    white_agent_url="http://localhost:9002",
    results_dir="./my_custom_results"  # Custom directory
)
```

### Max Tokens (in [green_agent/a2a_evaluator.py](green_agent/a2a_evaluator.py:116)):
```python
max_tokens=800  # Adjust based on needs
```

**Recommendations**:
- **Brief evaluations**: 300-500 tokens
- **Standard evaluations**: 800 tokens (current)
- **Detailed evaluations**: 1000-1500 tokens

## Model Comparison

| Model | Quality | Speed | Cost | JSON Reliability |
|-------|---------|-------|------|------------------|
| **GPT-4o-mini** (new) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰 | ✅ Excellent |
| Mistral-7B (old) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰 | ⚠️ Sometimes fails |
| GPT-4 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 💰💰💰💰 | ✅ Excellent |
| Claude-3-Sonnet | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰💰💰 | ✅ Excellent |

## Testing

```bash
# Run test script
conda run -n aipolicy python test_llm_judge.py

# Expected output:
# - Rule-based tests pass
# - LLM judge with GPT-4o-mini works
# - No errors
```

## Example Evaluation Session

```bash
$ conda run -n aipolicy python main.py launch --llm-judge

🚀 Launching AIPolicyBench A2A Evaluation
================================================================================
📗 Launching green agent (evaluator)...
✅ Green agent is ready at http://localhost:9001

📄 Launching white agent (RAG system)...
✅ Green agent is ready at http://localhost:9002

📤 Sending evaluation task to green agent...
⏳ Running evaluation (this may take a while)...

================================================================================
Starting evaluation of 7 queries
================================================================================

Results will be saved to: ./results/localhost_9002/20250113_211500

Query 1/7:  14%|█▍        | 1/7 [00:04<00:24, 4.1s/query] Correct: 1, Miss: 0, Halluc: 0
[Query 1] Are there datasets available for AI safety research?
[Ground Truth] yes
[White Agent Response]
Yes, there are several datasets available...

[Evaluating with LLM-as-a-judge]
[Judgment] ✅ CORRECT
[Confidence] 0.95
[Reasoning] The response correctly states...
[Saved] ./results/localhost_9002/20250113_211500/query_001.json

... (continues for all queries)

================================================================================
Evaluation complete: 5/7 correct (71.43%)
Correct: 5 (71.43%)
Miss: 1 (14.29%)
Hallucination: 1 (14.29%)
Factuality Rate: 85.71%
================================================================================

Summary saved to: ./results/localhost_9002/20250113_211500/summary.json
Statistics saved to: ./results/localhost_9002/20250113_211500/statistics.txt

✅ Evaluation complete!
```

## Analyzing Results

### View Summary:
```bash
cat ./results/localhost_9002/20250113_211500/statistics.txt
```

### Analyze with Python:
```python
import json

# Load summary
with open('./results/localhost_9002/20250113_211500/summary.json') as f:
    summary = json.load(f)

print(f"Total: {summary['statistics']['total']}")
print(f"Correct Rate: {summary['statistics']['correct_rate']:.2f}%")

# Analyze individual results
for result in summary['results']:
    if result['evaluation_result'] == 'hallucination':
        print(f"Hallucination in Query {result['query_id']}: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Reasoning: {result['reasoning']}")
        print()
```

### Compare Evaluations:
```bash
# List all evaluations
ls -lt ./results/localhost_9002/

# Compare two evaluations
diff ./results/localhost_9002/20250113_211500/statistics.txt \
     ./results/localhost_9002/20250113_214500/statistics.txt
```

## Troubleshooting

### Issue: No Results Directory Created

**Solution**: Check write permissions
```bash
chmod +w .
mkdir -p ./results
```

### Issue: Progress Bar Not Showing

**Solution**: Ensure tqdm is installed
```bash
conda run -n aipolicy pip install tqdm
```

### Issue: GPT-4o-mini Errors

**Solution**: Check OpenRouter API key and credits
```bash
# Verify API key is set
echo $DEEPSEEK_API_KEY

# Check credits at: https://openrouter.ai/credits
```

## Summary

✅ **Model**: Switched to GPT-4o-mini for better quality
✅ **Progress**: Added tqdm progress bar with live stats
✅ **Logging**: Detailed logging for each evaluation
✅ **Results**: Automatic saving to `./results/<agent>/<timestamp>/`
✅ **Files**: Individual JSON + summary JSON + statistics TXT
✅ **Tested**: All functionality verified and working

The LLM-as-a-judge evaluation is now production-ready with comprehensive logging and results tracking! 🎉
