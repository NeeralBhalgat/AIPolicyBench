# LLM-as-a-Judge Evaluation Guide

## Overview

AIPolicyBench now supports **LLM-as-a-Judge** evaluation in addition to rule-based evaluation. This provides more nuanced and semantically-aware evaluation of RAG agent responses.

## Evaluation Methods Comparison

| Feature | Rule-Based | LLM-as-a-Judge |
|---------|------------|----------------|
| **Speed** | ⚡ Fast | 🐢 Slower (requires LLM calls) |
| **Cost** | 💰 Free | 💰💰 Requires API credits |
| **Accuracy** | ✓ Good for exact matches | ✓✓ Better for semantic matching |
| **Reasoning** | ❌ No explanation | ✅ Provides reasoning |
| **Confidence** | ❌ Binary | ✅ Confidence scores |
| **Use Case** | Quick validation | Nuanced evaluation |

## How It Works

### Rule-Based Evaluation
```
Question: "What datasets for AI safety?"
Ground Truth: "yes"
Response: "Yes, there are several datasets for AI safety..."
Result: ✅ CORRECT (contains "yes")
```

### LLM-as-a-Judge Evaluation
```
Question: "What datasets for AI safety?"
Ground Truth: "yes"
Response: "There are several AI safety datasets available..."

LLM Judge analyzes:
- Does the response answer the question correctly?
- Does it contain the ground truth information?
- Is it factually accurate?

Result: ✅ CORRECT
Confidence: 0.95
Reasoning: "Response correctly affirms the existence of AI safety datasets"
```

## Usage

### 1. Via Command Line (Launcher)

```bash
# Use LLM judge
conda run -n aipolicy python main.py launch --llm-judge

# Use rule-based (default)
conda run -n aipolicy python main.py launch
```

### 2. Via Python API

```python
import asyncio
from launcher import launch_evaluation

# With LLM judge
asyncio.run(launch_evaluation(
    queries_file="data/predefined_queries.json",
    use_llm_judge=True  # Enable LLM judge
))

# With rule-based (default)
asyncio.run(launch_evaluation(
    queries_file="data/predefined_queries.json",
    use_llm_judge=False
))
```

### 3. Via A2A Message

When sending a task to the green agent:

```python
import asyncio
from utils import a2a_client

async def evaluate_with_llm_judge():
    task = """
<white_agent_url>http://localhost:9002</white_agent_url>
<queries_file>data/predefined_queries.json</queries_file>
<use_llm_judge>true</use_llm_judge>
"""

    response = await a2a_client.send_message(
        "http://localhost:9001",
        task
    )
    print(response)

asyncio.run(evaluate_with_llm_judge())
```

### 4. Direct Python Usage

```python
import asyncio
from green_agent.evaluation import LLMJudgeEvaluator

async def evaluate_response():
    # Initialize LLM judge
    evaluator = LLMJudgeEvaluator(
        provider="deepseek",  # or "openai", "anthropic"
        temperature=0.0
    )

    # Evaluate a single response
    result = await evaluator.evaluate(
        response="Yes, there are several datasets available for AI safety research.",
        ground_truth="yes",
        question="Are there datasets for AI safety?"
    )

    print(f"Result: {result['result']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")

asyncio.run(evaluate_response())
```

## Configuration

### Supported LLM Providers

The LLM judge supports multiple providers:

```python
from green_agent.evaluation import LLMJudgeEvaluator

# DeepSeek (default, cost-effective)
evaluator = LLMJudgeEvaluator(
    provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# OpenAI (GPT-4)
evaluator = LLMJudgeEvaluator(
    provider="openai",
    model="gpt-4-turbo-preview",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Anthropic (Claude)
evaluator = LLMJudgeEvaluator(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

### Environment Variables

Set up your API keys in `.env`:

```bash
# For DeepSeek (recommended, cost-effective)
DEEPSEEK_API_KEY=sk-your-deepseek-key

# For OpenAI
OPENAI_API_KEY=sk-your-openai-key

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

## Output Format

### Rule-Based Output
```json
{
  "result": "correct",
  "method": "Rule-based",
  "ground_truth": "yes",
  "match_found": true,
  "is_uncertain": false
}
```

### LLM Judge Output
```json
{
  "result": "correct",
  "method": "LLM-as-a-judge",
  "provider": "deepseek",
  "ground_truth": "yes",
  "confidence": 0.95,
  "reasoning": "Response correctly affirms the existence of AI safety datasets",
  "raw_judgment": {
    "classification": "correct",
    "confidence": 0.95,
    "reasoning": "Response correctly affirms the existence of AI safety datasets"
  }
}
```

## Classification Categories

Both methods classify responses into three categories:

### 1. **Correct** ✅
- Response contains the ground truth
- Factually accurate answer
- Directly addresses the question

**Example:**
```
Q: "Are there datasets for AI safety?"
GT: "yes"
Response: "Yes, there are several datasets..."
Result: ✅ CORRECT
```

### 2. **Miss** ⚠️
- Response expresses uncertainty
- Admits not knowing
- Safer than hallucinating

**Example:**
```
Q: "Who leads the evaluation team?"
GT: "Charles"
Response: "I'm not sure who leads that team."
Result: ⚠️ MISS
```

### 3. **Hallucination** ❌
- Response attempts an answer
- But provides incorrect information
- Factually wrong

**Example:**
```
Q: "Who leads the evaluation team?"
GT: "Charles"
Response: "John leads the evaluation team."
Result: ❌ HALLUCINATION
```

## Evaluation Metrics

### Correct Rate
```
Correct Rate = (Correct / Total) × 100%
```
Higher is better.

### Hallucination Rate
```
Hallucination Rate = (Hallucination / Total) × 100%
```
Lower is better.

### Miss Rate
```
Miss Rate = (Miss / Total) × 100%
```
Moderate is acceptable (uncertainty is safer than hallucination).

### Factuality Rate
```
Factuality Rate = Correct Rate + Miss Rate
                = 100% - Hallucination Rate
```
Higher is better. This rewards both correct answers and honest uncertainty.

## Example Results Comparison

### Rule-Based Evaluation
```
📊 Results Summary:
- Total Queries: 7
- Correct: 4 (57.14%)
- Miss: 2 (28.57%)
- Hallucination: 1 (14.29%)
- Factuality Rate: 85.71%

Evaluation Method: Rule-based
```

### LLM Judge Evaluation
```
📊 Results Summary:
- Total Queries: 7
- Correct: 5 (71.43%)
- Miss: 1 (14.29%)
- Hallucination: 1 (14.29%)
- Factuality Rate: 85.71%

Evaluation Method: LLM-as-a-judge
LLM Provider: deepseek

Detailed Results with Reasoning:
[
  {
    "query_id": 1,
    "result": "correct",
    "confidence": 0.95,
    "reasoning": "Response correctly identifies datasets with specific examples"
  },
  ...
]
```

## Performance Considerations

### Speed
- **Rule-based**: ~1 second for 7 queries
- **LLM judge**: ~30-60 seconds for 7 queries (depends on API)

### Cost
- **Rule-based**: Free
- **LLM judge with DeepSeek**: ~$0.001 per evaluation (very cheap)
- **LLM judge with GPT-4**: ~$0.01 per evaluation
- **LLM judge with Claude**: ~$0.015 per evaluation

### Accuracy
- **Rule-based**: Good for exact matches, may miss semantic equivalents
- **LLM judge**: Better at understanding semantic meaning and context

## Best Practices

### When to Use Rule-Based
✅ Quick validation during development
✅ Large-scale benchmarking (thousands of queries)
✅ Budget constraints
✅ Ground truth is simple (yes/no, names, numbers)

### When to Use LLM Judge
✅ Final evaluation for publication
✅ Nuanced questions requiring semantic understanding
✅ Complex ground truths
✅ Need for confidence scores and reasoning
✅ Debugging why responses fail

### Hybrid Approach
Use both methods and compare:

```bash
# Run with rule-based
conda run -n aipolicy python main.py launch > results_rulebased.txt

# Run with LLM judge
conda run -n aipolicy python main.py launch --llm-judge > results_llmjudge.txt

# Compare results
diff results_rulebased.txt results_llmjudge.txt
```

## Troubleshooting

### Issue: LLM Judge Returns Errors

**Symptom**: All evaluations return `"result": "error"`

**Solution**:
1. Check API key is set: `echo $DEEPSEEK_API_KEY`
2. Verify API credits available
3. Check network connectivity
4. Try different provider

### Issue: Low Confidence Scores

**Symptom**: Confidence scores consistently < 0.5

**Solution**:
1. Review ground truth quality - may be too vague
2. Check if questions are ambiguous
3. Try different LLM provider
4. Lower temperature (currently 0.0)

### Issue: Slow Evaluation

**Symptom**: Takes too long to complete

**Solution**:
1. Use rule-based for quick tests
2. Use faster LLM provider (DeepSeek is fastest)
3. Reduce number of queries during testing
4. Consider batch evaluation optimization

## Advanced: Custom LLM Judge

You can customize the LLM judge prompt by extending the class:

```python
from green_agent.evaluation import LLMJudgeEvaluator

class CustomJudge(LLMJudgeEvaluator):
    def _create_judge_prompt(self, question, response, ground_truth, context=None):
        # Your custom prompt logic
        return f"""Custom evaluation prompt...
        Question: {question}
        Response: {response}
        Ground Truth: {ground_truth}
        ...
        """

# Use custom judge
evaluator = CustomJudge(provider="deepseek")
```

## Conclusion

LLM-as-a-judge provides powerful, semantically-aware evaluation for RAG systems. While slower and more expensive than rule-based evaluation, it offers:

- ✅ Better semantic understanding
- ✅ Confidence scores
- ✅ Detailed reasoning
- ✅ More accurate for complex queries

Choose the right evaluation method based on your needs:
- **Development**: Rule-based
- **Final evaluation**: LLM judge
- **Best of both**: Run both and compare!
