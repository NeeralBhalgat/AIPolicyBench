# AIPolicyBench - Quick Reference Guide

## Project Structure at a Glance

```
AIPolicyBench/
├── Core Components
│   ├── safety_datasets_rag.py       <- RAG pipeline (retrieval, augmentation, generation)
│   ├── simple_vector_db.py          <- TF-IDF vector database
│   ├── green_agent/
│   │   ├── agent.py                 <- Query interface & evaluation runner
│   │   └── evaluation.py             <- Rule-based & LLM-judge evaluators
│   └── white_agent/agent.py         <- Task executor (stub)
│
├── Utilities
│   ├── utils/llm_client.py          <- Multi-provider LLM wrapper
│   └── scripts/parse_pdfs_to_json.py <- PDF document parser
│
├── Data
│   ├── data/safety_datasets.json    <- 183 policy document chunks
│   ├── data/predefined_queries.json <- Ground truth Q&A pairs
│   ├── docs/                        <- Source PDF documents
│   └── vector_db/                   <- Serialized TF-IDF database
│
├── Agent-to-Agent (A2A) Framework
│   └── agentify-example-tau-bench/
│       ├── main.py                  <- CLI entry point
│       ├── src/launcher.py          <- Evaluation coordinator
│       ├── src/green_agent/agent.py <- Assessment manager
│       ├── src/white_agent/agent.py <- Target agent
│       └── src/my_util/my_a2a.py    <- A2A communication
│
└── Configuration
    ├── requirements.txt             <- Python dependencies
    ├── .env                         <- API keys (not in repo)
    └── env.example                  <- API key template
```

## What Each Component Does

| File | Purpose | Key Classes |
|------|---------|-------------|
| `safety_datasets_rag.py` | Implements RAG pipeline | `SafetyDatasetsRAG` |
| `simple_vector_db.py` | TF-IDF semantic search | `SimpleTFIDFVectorDB` |
| `green_agent/agent.py` | Query evaluation interface | `PredefinedQueryInterface` |
| `green_agent/evaluation.py` | Response evaluation | `RuleBasedEvaluator`, `LLMJudgeEvaluator` |
| `utils/llm_client.py` | LLM provider abstraction | `LLMClient`, `DeepSeekClient`, `OpenAIClient` |
| `white_agent/agent.py` | Future task executor | Stub/template |
| `scripts/parse_pdfs_to_json.py` | PDF to JSON converter | Helper functions |

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp env.example .env
# Edit .env with your API keys
```

### 2. Run Query Evaluation
```bash
# Interactive mode
python green_agent/agent.py

# Evaluate all queries (rule-based)
python green_agent/agent.py --all

# Single query with LLM judge
python green_agent/agent.py --query_id 1 --use_llm_judge

# All queries with OpenAI
python green_agent/agent.py --all --use_llm_judge --llm_provider openai
```

### 3. Build Vector Database
```bash
# From existing JSON
python simple_vector_db.py --json_file data/safety_datasets.json --save

# After adding new PDFs
python scripts/parse_pdfs_to_json.py  # Parse PDFs
python simple_vector_db.py --json_file data/safety_datasets.json --save  # Rebuild DB
```

### 4. Run A2A Agent Evaluation
```bash
cd agentify-example-tau-bench
uv run python main.py launch  # Start both agents and evaluate
```

## Core Workflows

### RAG Pipeline (4 Steps)
```
Query Input
  ↓
1. RETRIEVAL: SimpleTFIDFVectorDB.search() → Top-k chunks
  ↓
2. AUGMENTATION: Prepare context with metadata
  ↓
3. GENERATION: LLMClient.generate_response() → Answer
  ↓
Response Output
```

### Evaluation Methods
```
Rule-Based Evaluator:
  Response → Check ground truth presence → {correct, miss, hallucination}

LLM-Judge Evaluator:
  Response → Send to LLM → {classification, confidence, reasoning}
```

### A2A Agent Communication
```
Green Agent (port 9001) ←→ A2A Messages ←→ White Agent (port 9002)
  (Evaluator)                                (Task Executor)
```

## Key Methods & APIs

### SafetyDatasetsRAG
```python
rag = SafetyDatasetsRAG()
rag.load_vector_db()                          # Load database
results = rag.retrieve(query, top_k=5)        # Get relevant chunks
response = rag.complete_rag_query(query)      # Full RAG pipeline
```

### SimpleTFIDFVectorDB
```python
db = SimpleTFIDFVectorDB()
db.add_documents(docs, metadatas, ids)        # Add documents
results = db.search(query, top_k=5)           # Semantic search
db.save(filepath)                             # Persist
db.load(filepath)                             # Load
```

### Evaluation
```python
# Rule-based
rule_eval = RuleBasedEvaluator()
result = rule_eval.evaluate(response, ground_truth)
batch_result = rule_eval.evaluate_batch(responses, truths)

# LLM Judge
llm_eval = LLMJudgeEvaluator(provider="deepseek")
result = await llm_eval.evaluate(response, ground_truth, question)
batch = await llm_eval.evaluate_batch(responses, truths, questions)
```

### LLMClient
```python
client = LLMClient(provider="deepseek")
response = await client.generate_response(prompt)
response = await client.generate_policy_response(question, context)
```

### Query Interface
```python
interface = PredefinedQueryInterface()
interface.initialize()
result = asyncio.run(interface.evaluate_query(query_id=1))
all_results = asyncio.run(interface.evaluate_all_queries())
```

## Configuration

### Environment Variables
```bash
# Required for LLM generation
DEEPSEEK_API_KEY=...        # DeepSeek or OpenRouter (sk-or-*)
OPENAI_API_KEY=...          # OpenAI GPT models
ANTHROPIC_API_KEY=...       # Anthropic Claude models

# Optional paths
VECTOR_DB_PATH=./vector_db/safety_datasets_tfidf_db.pkl
QUERIES_FILE=data/predefined_queries.json
```

### Data Files
- **safety_datasets.json**: 183 chunks from policy documents (auto-generated)
- **predefined_queries.json**: 7+ Q&A pairs with ground truth
- **safety_datasets_tfidf_db.pkl**: Pickled vector database (auto-generated)

## Evaluation Metrics

```
Correct Rate = (Correct / Total) × 100%
Hallucination Rate = (Hallucinations / Total) × 100%
Miss Rate = (Misses / Total) × 100%
Factuality Rate = Correct% - Hallucination% + (0.5 × Miss%)
```

## Extending the System

### Add a New Query
Edit `data/predefined_queries.json`:
```json
{
  "id": 8,
  "query": "Your question?",
  "ground_truth": "Expected answer"
}
```

### Add New Documents
```bash
cp your_document.pdf docs/
python scripts/parse_pdfs_to_json.py
python simple_vector_db.py --json_file data/safety_datasets.json --save
```

### Add LLM Provider
1. Create client class in `utils/llm_client.py`
2. Inherit from `BaseLLMClient`
3. Implement `generate_response()` and `generate_chat_response()`
4. Register in `LLMClient.__init__()`

### Custom Evaluation
Extend `green_agent/evaluation.py`:
```python
class CustomEvaluator:
    def evaluate(self, response, ground_truth):
        # Your logic here
        return {"result": "...", "confidence": 0.9}
```

## Common Commands

```bash
# Development
python -m pytest tests/
black .                                    # Format code
flake8 green_agent/ white_agent/ utils/   # Lint code

# Maintenance
python scripts/parse_pdfs_to_json.py --chunk_size 1000 --overlap 200
python simple_vector_db.py --json_file data/safety_datasets.json --save
python green_agent/evaluation.py           # Run evaluation tests

# Debugging
python -c "from safety_datasets_rag import SafetyDatasetsRAG; rag = SafetyDatasetsRAG(); print('RAG OK')"
python -c "from simple_vector_db import SimpleTFIDFVectorDB; db = SimpleTFIDFVectorDB(); print('DB OK')"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key not working | Update .env, ensure key format is correct |
| Vector DB not found | Run `python simple_vector_db.py --json_file data/safety_datasets.json --save` |
| Queries not loading | Check `data/predefined_queries.json` exists and is valid JSON |
| Import errors | Run `pip install -r requirements.txt` |
| A2A agent not ready | Check port availability (9001, 9002) and firewall rules |

## Performance Notes

- **TF-IDF Search**: Fast (<100ms for 183 chunks)
- **LLM Inference**: Slow (1-5 seconds per query)
- **Evaluation**: Rule-based is instant, LLM-judge requires API call
- **Memory**: ~50MB for vector database + models

## Key Insights

1. **Two-tier Evaluation**: Rule-based for speed, LLM-judge for accuracy
2. **Flexible LLM Support**: Switch providers with one parameter
3. **Async-first Design**: All I/O operations are non-blocking
4. **Modular Architecture**: Easy to extend with new evaluators/providers
5. **A2A Protocol**: Standardized inter-agent communication (Tau-Bench example)

---

For comprehensive documentation, see `CODEBASE_OVERVIEW.md`
