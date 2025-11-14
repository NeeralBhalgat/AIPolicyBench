# AIPolicyBench - Complete File Structure & Key Files

## Directory Tree with Descriptions

```
/home/momoway/AIPolicyBench/
│
├── CODEBASE_OVERVIEW.md              ← Comprehensive architecture documentation
├── QUICK_REFERENCE.md                ← Quick start and API reference
├── FILE_STRUCTURE.md                 ← This file
├── README.md                         ← Project overview & setup guide
│
├── requirements.txt                  ← Python dependencies
├── env.example                       ← API key template
├── .env                              ← API keys (git-ignored)
│
├── green_agent/                      ← EVALUATION AGENT
│   ├── agent.py                      ← Query interface & runner
│   │   └─ Class: PredefinedQueryInterface
│   │   └─ Methods: evaluate_query(), evaluate_all_queries(), interactive_mode()
│   └── evaluation.py                 ← Response evaluation framework
│       ├─ Class: RuleBasedEvaluator
│       │  └─ Methods: evaluate(), evaluate_batch()
│       └─ Class: LLMJudgeEvaluator
│           └─ Methods: evaluate(), evaluate_batch()
│
├── white_agent/                      ← TASK EXECUTOR AGENT (stub)
│   └── agent.py                      ← LLM client template
│
├── utils/                            ← UTILITIES
│   └── llm_client.py                 ← Multi-provider LLM wrapper
│       ├─ Class: BaseLLMClient (abstract)
│       ├─ Class: DeepSeekClient
│       ├─ Class: OpenAIClient
│       ├─ Class: AnthropicClient
│       ├─ Class: LocalLLMClient
│       └─ Class: LLMClient (unified interface)
│
├── safety_datasets_rag.py            ← RAG PIPELINE
│   └─ Class: SafetyDatasetsRAG
│      └─ Methods: load_vector_db(), retrieve(), augment(), generate(), complete_rag_query()
│
├── simple_vector_db.py               ← VECTOR DATABASE
│   └─ Class: SimpleTFIDFVectorDB
│      └─ Methods: add_documents(), search(), save(), load()
│
├── scripts/                          ← UTILITY SCRIPTS
│   ├── parse_pdfs_to_json.py         ← PDF parsing & chunking
│   └── README.md                     ← Scripts documentation
│
├── data/                             ← DATA & DATASETS
│   ├── safety_datasets.json          ← 183 document chunks (auto-generated)
│   ├── predefined_queries.json       ← 7+ Q&A pairs with ground truth
│   └── safety_datasets.csv           ← Legacy CSV format
│
├── docs/                             ← SOURCE DOCUMENTS
│   ├── 1 Americas-AI-Action-Plan.pdf
│   └── 5 Artificial Intelligence Risk Management Framework (AI RMF 1.0).pdf
│
├── vector_db/                        ← VECTOR DATABASE (SERIALIZED)
│   └── safety_datasets_tfidf_db.pkl  ← Pickled TF-IDF database
│
├── agentify-example-tau-bench/       ← A2A AGENT FRAMEWORK
│   ├── main.py                       ← CLI entry point (Typer)
│   ├── README.md                     ← A2A setup guide
│   └── src/
│       ├── launcher.py               ← Evaluation coordinator
│       │   └─ Function: launch_evaluation() - starts agents and coordinates
│       ├── green_agent/
│       │   ├── agent.py              ← Assessment manager
│       │   │   └─ Class: TauGreenAgentExecutor
│       │   │      └─ Methods: execute(), ask_agent_to_solve()
│       │   └── __init__.py
│       ├── white_agent/
│       │   ├── agent.py              ← Target agent (task executor)
│       │   │   └─ Class: GeneralWhiteAgentExecutor
│       │   │      └─ Methods: execute()
│       │   └── __init__.py
│       ├── my_util/
│       │   ├── my_a2a.py             ← A2A communication helpers
│       │   │   └─ Functions: get_agent_card(), wait_agent_ready(), send_message()
│       │   └── __init__.py
│       └── __init__.py
│
└── .git/                             ← Git repository
```

---

## Core Files - Detailed Description

### RAG Pipeline

**File**: `/home/momoway/AIPolicyBench/safety_datasets_rag.py`
- **Purpose**: Implements the 4-step RAG pipeline
- **Size**: ~400 lines
- **Key Class**: `SafetyDatasetsRAG`
- **Methods**:
  - `load_vector_db()`: Load TF-IDF database from disk
  - `retrieve(query, top_k=5)`: Find relevant document chunks
  - `augment(items)`: Prepare context with metadata
  - `generate(context, use_llm=True)`: Generate response with LLM
  - `complete_rag_query(query, top_k)`: Execute full pipeline
- **Dependencies**: SimpleVectorDB, LLMClient, asyncio

### Vector Database

**File**: `/home/momoway/AIPolicyBench/simple_vector_db.py`
- **Purpose**: TF-IDF-based semantic search
- **Size**: ~350 lines
- **Key Class**: `SimpleTFIDFVectorDB`
- **Methods**:
  - `add_documents(docs, metadatas, ids)`: Ingest documents
  - `search(query, top_k=5)`: Semantic search with cosine similarity
  - `save(filepath)`: Persist to pickle file
  - `load(filepath)`: Load from pickle file
- **Configuration**:
  - Max features: 10,000
  - N-gram range: (1, 2)
  - Similarity metric: Cosine
- **Data**: 183 chunks with metadata

### Evaluation Framework

**File**: `/home/momoway/AIPolicyBench/green_agent/evaluation.py`
- **Purpose**: Evaluate responses against ground truth
- **Size**: ~640 lines
- **Key Classes**:
  - `RuleBasedEvaluator`: Fast substring + uncertainty detection
    - Methods: `evaluate()`, `evaluate_batch()`
  - `LLMJudgeEvaluator`: Semantic evaluation by LLM
    - Methods: `evaluate()`, `evaluate_batch()`
- **Classifications**: correct, miss, hallucination
- **Metrics**: Correct Rate, Hallucination Rate, Miss Rate, Factuality Rate
- **Uncertainty Phrases** (12 variants detected)

### Green Agent (Query Interface)

**File**: `/home/momoway/AIPolicyBench/green_agent/agent.py`
- **Purpose**: Query evaluation and management interface
- **Size**: ~406 lines
- **Key Class**: `PredefinedQueryInterface`
- **Methods**:
  - `initialize()`: Load RAG and evaluator
  - `evaluate_query(query_id, top_k=5)`: Single query evaluation
  - `evaluate_all_queries(top_k=5)`: Batch evaluation
  - `interactive_mode()`: Interactive CLI
  - `show_queries()`: Display available queries
- **Features**:
  - Three evaluation modes: interactive, batch, single
  - Support for rule-based and LLM-judge evaluation
  - Rich output formatting
- **CLI Options**:
  - `--query_id`: Single query ID
  - `--all`: Batch mode
  - `--use_llm_judge`: Use LLM evaluator
  - `--llm_provider`: DeepSeek, OpenAI, or Anthropic

### LLM Client

**File**: `/home/momoway/AIPolicyBench/utils/llm_client.py`
- **Purpose**: Unified LLM provider interface
- **Size**: ~435 lines
- **Key Classes**:
  - `BaseLLMClient` (abstract)
  - `DeepSeekClient`: Direct API + OpenRouter support
  - `OpenAIClient`: GPT models
  - `AnthropicClient`: Claude models
  - `LocalLLMClient`: Transformers-based
  - `LLMClient`: Unified wrapper
- **Features**:
  - Automatic API key detection from .env
  - OpenRouter auto-detection (sk-or-* keys)
  - Async/await support
  - Policy-specific methods (generate_policy_response, etc.)

### PDF Parser

**File**: `/home/momoway/AIPolicyBench/scripts/parse_pdfs_to_json.py`
- **Purpose**: Parse PDF documents into semantic chunks
- **Features**:
  - Intelligent chunking (paragraph + sentence level)
  - Configurable chunk size and overlap
  - JSON output with metadata
- **Output**: `data/safety_datasets.json`
- **Parameters**:
  - `--chunk_size`: Chunk size in characters (default: 1500)
  - `--overlap`: Overlap between chunks (default: 300)

---

## Data Files

### Safety Datasets (data/safety_datasets.json)
- **Format**: JSON array of chunks
- **Size**: ~264 KB
- **Records**: 183 chunks
- **Schema**:
  ```json
  {
    "id": "source_chunk_0",
    "text": "Source: filename.pdf\n\nContent..."
  }
  ```
- **Sources**:
  - Americas AI Action Plan (74 chunks)
  - AI Risk Management Framework (109 chunks)

### Predefined Queries (data/predefined_queries.json)
- **Format**: JSON object with queries array
- **Records**: 7+ query pairs
- **Schema**:
  ```json
  {
    "id": 1,
    "query": "Question?",
    "ground_truth": "Expected answer",
    "keyword": "search term",
    "quote": "Source quote"
  }
  ```
- **Purpose**: Benchmark testing with known answers

### Vector Database (vector_db/safety_datasets_tfidf_db.pkl)
- **Format**: Pickled Python object
- **Size**: ~2-5 MB
- **Contains**:
  - TfidfVectorizer instance
  - TF-IDF matrix (sparse CSR format)
  - Document texts and IDs
  - Metadata

---

## A2A Agent Framework

### Launcher (agentify-example-tau-bench/src/launcher.py)
- **Purpose**: Coordinate multi-agent evaluation
- **Function**: `launch_evaluation()`
- **Workflow**:
  1. Start green agent (port 9001)
  2. Start white agent (port 9002)
  3. Send task description to green agent
  4. Wait for evaluation
  5. Terminate agents

### Green Agent - Agentify (agentify-example-tau-bench/src/green_agent/agent.py)
- **Purpose**: Assessment manager (A2A server)
- **Class**: `TauGreenAgentExecutor`
- **Workflow**:
  1. Receive task with environment config
  2. Initialize Tau-Bench environment
  3. Send observations to white agent (A2A)
  4. Receive actions and execute in environment
  5. Report metrics
- **Port**: 9001

### White Agent - Agentify (agentify-example-tau-bench/src/white_agent/agent.py)
- **Purpose**: Task executor (A2A server)
- **Class**: `GeneralWhiteAgentExecutor`
- **Features**:
  - Uses LiteLLM (unified LLM interface)
  - Maintains conversation context
  - Returns responses via A2A event queue
- **Port**: 9002

### A2A Utilities (agentify-example-tau-bench/src/my_util/my_a2a.py)
- **Purpose**: A2A protocol helpers
- **Functions**:
  - `get_agent_card(url)`: Fetch agent metadata
  - `wait_agent_ready(url, timeout)`: Poll for readiness
  - `send_message(url, message, task_id, context_id)`: Send message
- **Client**: `httpx` async HTTP client
- **Protocol**: Agent card registration + message routing

---

## Configuration Files

### env.example
- Template for API key configuration
- Keys: DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
- Location: `/home/momoway/AIPolicyBench/env.example`

### .env (not in repo)
- Actual API keys (git-ignored)
- Loaded by dotenv before execution
- Required for LLM-based operations

### requirements.txt
- Core dependencies: pandas, PyPDF2, scikit-learn
- LLM integration: openai>=1.6.0, langchain
- Web framework: fastapi, uvicorn
- Utilities: python-dotenv, pydantic

---

## Key Absolute File Paths

```
/home/momoway/AIPolicyBench/safety_datasets_rag.py
/home/momoway/AIPolicyBench/simple_vector_db.py
/home/momoway/AIPolicyBench/green_agent/agent.py
/home/momoway/AIPolicyBench/green_agent/evaluation.py
/home/momoway/AIPolicyBench/white_agent/agent.py
/home/momoway/AIPolicyBench/utils/llm_client.py
/home/momoway/AIPolicyBench/scripts/parse_pdfs_to_json.py
/home/momoway/AIPolicyBench/data/safety_datasets.json
/home/momoway/AIPolicyBench/data/predefined_queries.json
/home/momoway/AIPolicyBench/vector_db/safety_datasets_tfidf_db.pkl
/home/momoway/AIPolicyBench/agentify-example-tau-bench/main.py
/home/momoway/AIPolicyBench/agentify-example-tau-bench/src/launcher.py
/home/momoway/AIPolicyBench/agentify-example-tau-bench/src/green_agent/agent.py
/home/momoway/AIPolicyBench/agentify-example-tau-bench/src/white_agent/agent.py
/home/momoway/AIPolicyBench/agentify-example-tau-bench/src/my_util/my_a2a.py
```

---

## Import Structure

### Main RAG Pipeline
```python
from safety_datasets_rag import SafetyDatasetsRAG
from simple_vector_db import SimpleTFIDFVectorDB
from utils.llm_client import LLMClient
```

### Evaluation
```python
from green_agent.evaluation import RuleBasedEvaluator, LLMJudgeEvaluator
from green_agent.agent import PredefinedQueryInterface
```

### A2A Framework
```python
from src.launcher import launch_evaluation
from src.my_util.my_a2a import get_agent_card, wait_agent_ready, send_message
```

---

## File Statistics

| Category | Files | Lines | Size |
|----------|-------|-------|------|
| Core RAG | 2 | ~750 | ~30 KB |
| Agents | 2 | ~400 | ~15 KB |
| Evaluation | 1 | ~640 | ~25 KB |
| Utilities | 1 | ~435 | ~17 KB |
| Scripts | 1 | ~200 | ~8 KB |
| A2A Framework | 5 | ~600 | ~25 KB |
| **Total** | **12** | **~3,025** | **~120 KB** |

---

## Git Repository Information

- **Remote**: https://github.com/NeeralBhalgat/AIPolicyBench
- **Current Branch**: main
- **Status**: Clean (all changes committed)
- **Recent Commits**:
  1. fde4030 - Change names
  2. bbf5010 - Merge branch 'main'
  3. f025099 - renamed changes
  4. 8e78fea - Add new questions and docs
  5. 4124748 - Add llm-as-a-judge

---

End of File Structure Documentation
