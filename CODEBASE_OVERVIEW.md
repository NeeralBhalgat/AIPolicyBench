# AIPolicyBench - Comprehensive Codebase Overview

## Project Summary

AIPolicyBench is a sophisticated **Retrieval-Augmented Generation (RAG) system** designed for AI policy research and agent evaluation. It combines a safety dataset knowledge base with a dual-agent framework (White Agent for task execution, Green Agent for evaluation) to assess and benchmark AI agent responses against ground truth answers.

---

## 1. CORE ARCHITECTURE & COMPONENTS

### 1.1 Main System Architecture

```
AIPolicyBench/
├── Core RAG System
│   ├── safety_datasets_rag.py          (RAG pipeline: retrieval, augmentation, generation)
│   ├── simple_vector_db.py             (TF-IDF vector database implementation)
│   └── green_agent/agent.py            (Green Agent - query interface)
│
├── Evaluation Framework
│   ├── green_agent/evaluation.py       (Rule-based & LLM-as-a-judge evaluators)
│   └── Query validation & metrics
│
├── LLM Integration
│   ├── utils/llm_client.py             (Unified LLM client wrapper)
│   └── Support for: DeepSeek, OpenAI, Anthropic, Local models
│
├── Agent-to-Agent (A2A) System
│   ├── agentify-example-tau-bench/     (Agentified benchmark framework)
│   └── Tau-Bench integration
│
├── Data & Resources
│   ├── data/safety_datasets.json       (183 chunks from policy documents)
│   ├── data/predefined_queries.json    (Ground truth queries for evaluation)
│   └── docs/                           (Source PDF documents)
│
└── Utilities
    ├── scripts/parse_pdfs_to_json.py   (PDF parsing & chunking)
    ├── vector_db/                      (Serialized TF-IDF database)
    └── requirements.txt                (Dependencies)
```

---

## 2. KEY COMPONENTS IN DETAIL

### 2.1 RAG Pipeline (safety_datasets_rag.py)

**Purpose**: Core retrieval-augmented generation system for policy question answering

**Flow**:
```
Query Input
    ↓
Step 1: RETRIEVAL → Find relevant datasets from 183 chunks (TF-IDF)
    ↓
Step 2: AUGMENTATION → Prepare rich context with metadata
    ↓
Step 3: GENERATION → DeepSeek LLM creates natural response
    ↓
Step 4: RESPONSE → Return actionable policy guidance
```

**Key Features**:
- Query retrieval with configurable top_k (default: 5)
- Context augmentation with source metadata
- LLM-based response generation
- Handles both direct DeepSeek API and OpenRouter endpoints
- Async/await support for concurrent processing

**Main Methods**:
```python
load_vector_db()                    # Load TF-IDF database
retrieve(query, top_k=5)            # Step 1: Retrieval
augment(retrieved_items)            # Step 2: Augmentation
generate(context, use_llm=True)     # Step 3: Generation
complete_rag_query(query, top_k)    # Steps 1-4 combined
```

---

### 2.2 Vector Database (simple_vector_db.py)

**Purpose**: TF-IDF-based semantic search over policy documents

**Implementation Details**:
- **Method**: TF-IDF vectorization with cosine similarity
- **Features**: 10,000 max features, (1,2) n-gram range, English stop words
- **Similarity Metric**: Cosine similarity for ranking results
- **Storage**: Pickle serialization for fast disk I/O

**Key Methods**:
```python
add_documents(documents, metadatas, ids)    # Ingest documents
search(query, top_k=5)                      # Semantic search
save(filepath)                              # Persist to disk
load(filepath)                              # Load from disk
```

**Current Dataset**:
- **183 chunks** from 2 policy documents
- **Average chunk size**: 1,443 characters
- **Sources**:
  - Americas AI Action Plan (28 pages → 74 chunks)
  - AI Risk Management Framework (48 pages → 109 chunks)

---

### 2.3 Evaluation Framework (green_agent/evaluation.py)

**Purpose**: Assess agent responses against ground truth answers

**Two Evaluation Methods**:

#### A. Rule-Based Evaluator
- **Speed**: Fast (no API calls)
- **Mechanism**: Substring matching + uncertainty phrase detection
- **Classification**:
  - `correct`: Response contains ground truth
  - `miss`: Response expresses uncertainty ("I don't know", "I'm not sure")
  - `hallucination`: Response is factually incorrect
- **Use Case**: Fast validation, exact matches, budget-conscious

#### B. LLM-as-a-Judge Evaluator (Green Agent)
- **Speed**: Slower (requires LLM API)
- **Mechanism**: Semantic evaluation by trusted LLM
- **Classification**: Same as rule-based (correct/miss/hallucination)
- **Features**:
  - Understands semantic equivalence ("FDA" = "Food and Drug Administration")
  - Provides confidence scores (0-1)
  - Includes detailed reasoning for each judgment
  - Supports multiple LLM providers
- **Use Case**: Complex answers, paraphrasing, semantic understanding needed

**Metrics Calculated**:
```
Correct Rate = (Correct / Total) × 100%
Hallucination Rate = (Hallucinations / Total) × 100%
Miss Rate = (Misses / Total) × 100%
Factuality Rate = Correct Rate - Hallucination Rate + (c × Miss Rate)
                where c = miss_weight (default 0.5)
```

**Example**:
- 6 questions, 3 correct (50%), 1 hallucination (16.67%), 2 misses (33.33%)
- Factuality Rate = 50% - 16.67% + (0.5 × 33.33%) = **50%**

---

### 2.4 LLM Client Integration (utils/llm_client.py)

**Purpose**: Unified interface for multiple LLM providers

**Supported Providers**:
- **DeepSeek**: Direct API or OpenRouter (auto-detected by API key prefix)
- **OpenAI**: GPT-4, GPT-4-turbo, etc.
- **Anthropic**: Claude models
- **Local**: Transformers-based models

**Key Features**:
- Automatic API key detection from environment
- OpenRouter auto-detection (keys starting with `sk-or-`)
- Async/await support
- Consistent interface across providers

**Available Methods**:
```python
generate_response(prompt, **kwargs)           # Single prompt
generate_chat_response(messages, **kwargs)    # Multi-turn conversation
generate_policy_response(question, context)   # Policy-focused analysis
generate_summary(text, summary_type)          # Text summarization
generate_analysis(topic, context, type)       # Custom analysis types
```

---

### 2.5 Agent System (green_agent/agent.py & white_agent/agent.py)

#### Green Agent (Evaluator)
**File**: `/home/momoway/AIPolicyBench/green_agent/agent.py`

**Purpose**: Evaluation interface for predefined queries

**Key Class**: `PredefinedQueryInterface`
- Loads queries from JSON file
- Manages evaluation execution
- Supports interactive and batch modes
- Delegates to evaluators (rule-based or LLM-judge)

**Main Modes**:
1. **Interactive Mode**: User selects queries to evaluate
2. **Batch Mode**: Evaluate all queries at once
3. **Single Query Mode**: Evaluate specific query by ID

**Key Methods**:
```python
initialize()                                  # Load RAG and evaluator
evaluate_query(query_id, top_k=5)            # Evaluate single query
evaluate_all_queries(top_k=5)                # Batch evaluation
display_query_result(result)                 # Format output
display_all_results(batch_result)            # Display statistics
interactive_mode()                           # Run interactive CLI
```

#### White Agent (Task Executor)
**File**: `/home/momoway/AIPolicyBench/white_agent/agent.py`

**Purpose**: Placeholder for future agent implementation

**Current State**: Stub/template file showing LLM client usage patterns

---

## 3. AGENT-TO-AGENT (A2A) SYSTEM

### 3.1 Agentify Example Structure

**Location**: `/home/momoway/AIPolicyBench/agentify-example-tau-bench/`

**Purpose**: Implements standardized A2A communication for agent-based benchmarking

**Directory Structure**:
```
agentify-example-tau-bench/
├── main.py                        # CLI entry point (Typer commands)
├── src/
│   ├── launcher.py               # Evaluation coordinator
│   ├── green_agent/agent.py      # Assessment manager (A2A server)
│   ├── white_agent/agent.py      # Target agent (A2A server)
│   └── my_util/my_a2a.py         # A2A client utilities
└── README.md
```

### 3.2 A2A Communication Framework

**A2A Standard**: Agent-to-Agent protocol for inter-agent communication

**Key Components**:

#### A2A Client Utilities (my_util/my_a2a.py)
```python
async get_agent_card(url)                    # Fetch agent metadata
async wait_agent_ready(url, timeout=10)      # Wait for agent startup
async send_message(url, message, task_id, context_id)  # Send message
```

**Features**:
- Async HTTP communication (httpx)
- Agent card resolution (metadata discovery)
- Message envelope with roles, parts, IDs
- Context management for multi-turn conversations
- Unique message and request IDs (UUID)

#### Green Agent Implementation (agentify-example-tau-bench/src/green_agent/agent.py)
```python
TauGreenAgentExecutor
├── execute()          # Main evaluation loop
└── ask_agent_to_solve()  # Send tasks to white agent via A2A
```

**Workflow**:
1. Receives task description with environment config
2. Sets up Tau-Bench environment
3. Enters interaction loop with white agent:
   - Sends task/observation to white agent
   - Receives action response (tool calls or final response)
   - Executes action in environment
   - Provides result back to white agent
4. Reports metrics and success/failure

#### White Agent Implementation (agentify-example-tau-bench/src/white_agent/agent.py)
```python
GeneralWhiteAgentExecutor
├── execute()          # Handle incoming requests
└── Maintains message history per context_id
```

**Features**:
- Uses LiteLLM (unified LLM interface)
- Maintains conversation context per context_id
- Processes tool calling and action selection
- Returns responses via A2A event queue

#### Launcher/Coordinator (agentify-example-tau-bench/src/launcher.py)
```python
async launch_evaluation()
├── Start green agent (localhost:9001)
├── Start white agent (localhost:9002)
├── Send task description to green agent
├── Wait for evaluation completion
└── Terminate agents
```

**Execution Flow**:
```
User → Launch Evaluation
    ↓
Start Green Agent (A2A Server)
    ↓
Start White Agent (A2A Server)
    ↓
Send Task Config to Green Agent
    ↓
Green Agent ↔ White Agent (A2A Messages)
    ↓
Green Agent Evaluates White Agent Responses
    ↓
Report Results
```

### 3.3 A2A Message Format

**Message Structure**:
```python
Message(
    role=Role.user,                    # sender role
    parts=[Part(TextPart(text=message))],  # content
    message_id=uuid.uuid4().hex,       # unique ID
    task_id=task_id,                   # task identifier
    context_id=context_id              # conversation context
)
```

**Sending via A2A Client**:
```python
SendMessageRequest(
    id=request_id,
    params=MessageSendParams(message=message)
)
```

---

## 4. DATA STRUCTURES & FORMATS

### 4.1 Safety Datasets (data/safety_datasets.json)

**Structure**:
```json
[
  {
    "id": "Americas-AI-Action-Plan_chunk_0",
    "text": "Source: Americas-AI-Action-Plan.pdf\n\nChunk content..."
  },
  ...
]
```

**Generated by**: `scripts/parse_pdfs_to_json.py`
**Format**: Simple JSON with id and text fields
**Size**: 183 chunks, ~264 KB total

### 4.2 Predefined Queries (data/predefined_queries.json)

**Structure**:
```json
{
  "queries": [
    {
      "id": 1,
      "query": "What are the three pillars of America's AI Action Plan?",
      "ground_truth": "Innovation, infrastructure, and international diplomacy and security.",
      "keyword": "three pillars",
      "quote": "Full quote from source"
    },
    ...
  ]
}
```

**Current Dataset**: 7+ queries with ground truth answers
**Purpose**: Benchmark agent responses for accuracy
**Evaluation**: Both rule-based and LLM-as-a-judge methods

### 4.3 Vector Database (vector_db/safety_datasets_tfidf_db.pkl)

**Format**: Pickled Python object containing:
- TfidfVectorizer instance
- TF-IDF matrix (sparse CSR format)
- Document texts
- Document IDs and metadata

**Generated by**: `python simple_vector_db.py --json_file data/safety_datasets.json --save`

---

## 5. EXECUTION FLOWS

### 5.1 Query Evaluation Flow (Main AIPolicyBench)

```
User Input
    ↓
PredefinedQueryInterface.evaluate_query(query_id)
    ↓
SafetyDatasetsRAG.complete_rag_query()
    ├─ retrieve(query) → Find top-k relevant chunks
    ├─ augment() → Prepare context
    └─ generate() → LLM response
    ↓
Evaluator.evaluate() [rule-based or LLM-judge]
    ├─ Parse response
    ├─ Compare with ground truth
    └─ Return classification + confidence
    ↓
Display Results + Metrics
```

### 5.2 Agentify A2A Evaluation Flow

```
launch_evaluation()
    ↓
Start Green Agent (port 9001)
    ↓
Start White Agent (port 9002)
    ↓
Green Agent receives task description
    ↓
Initialize Tau-Bench environment
    ↓
For each step:
    ├─ Green Agent sends observation to White Agent (HTTP POST)
    ├─ White Agent processes & generates response (LLM call)
    ├─ Green Agent receives response
    ├─ Parse action from response
    ├─ Execute in environment
    └─ Get result
    ↓
Compare results to success criteria
    ↓
Report metrics + termination
```

---

## 6. COMMAND-LINE INTERFACES

### 6.1 Main Query Interface

```bash
# Interactive mode
python green_agent/agent.py

# Evaluate all queries (rule-based)
python green_agent/agent.py --all

# Evaluate single query
python green_agent/agent.py --query_id 1

# Use LLM-as-a-judge evaluation
python green_agent/agent.py --all --use_llm_judge --llm_provider deepseek

# With OpenAI
python green_agent/agent.py --all --use_llm_judge --llm_provider openai

# Custom options
python green_agent/agent.py --query_id 1 --top_k 10 --use_llm_judge
```

### 6.2 Vector Database Setup

```bash
# Build vector database from JSON
python simple_vector_db.py --json_file data/safety_datasets.json --save

# Load and test
python simple_vector_db.py --json_file data/safety_datasets.json --load
```

### 6.3 PDF Processing

```bash
# Parse PDFs with default settings (1500 chars, 300 overlap)
python scripts/parse_pdfs_to_json.py

# Custom chunk sizes
python scripts/parse_pdfs_to_json.py --chunk_size 1000 --overlap 200

# Rebuild vector database after parsing
python simple_vector_db.py --json_file data/safety_datasets.json --save
```

### 6.4 Agentify Evaluation

```bash
# Launch complete evaluation (green + white agents)
cd agentify-example-tau-bench
uv run python main.py launch

# Start agents individually
uv run python main.py green
uv run python main.py white
```

---

## 7. DEPENDENCIES & REQUIREMENTS

### Core Dependencies (requirements.txt)

**Data Processing**:
- pandas>=2.0.0
- PyPDF2>=3.0.0
- beautifulsoup4>=4.12.0

**Vector Database & ML**:
- scikit-learn>=1.3.0
- numpy>=1.24.0
- faiss-cpu>=1.7.0

**LLM Integration**:
- openai>=1.6.0
- langchain>=0.1.0

**Web Framework**:
- fastapi>=0.108.0
- uvicorn>=0.25.0

**Utilities**:
- python-dotenv>=1.0.0
- pydantic>=2.5.0

**Development**:
- pytest>=7.4.0
- black>=23.12.0

### Agentify-Specific Dependencies

From agentify-example-tau-bench:
- `a2a`: Agent-to-Agent protocol implementation
- `tau-bench`: Benchmark environment integration
- `litellm`: Unified LLM interface
- `typer`: CLI framework
- `uvicorn`: ASGI server

---

## 8. CONFIGURATION & ENVIRONMENT

### Environment Variables (.env)

```bash
# LLM API Keys
DEEPSEEK_API_KEY=...          # DeepSeek direct or OpenRouter (sk-or-*)
OPENAI_API_KEY=...            # OpenAI API key
ANTHROPIC_API_KEY=...         # Anthropic API key

# Optional paths
VECTOR_DB_PATH=./vector_db/safety_datasets_tfidf_db.pkl
QUERIES_FILE=data/predefined_queries.json
```

### Automatic API Key Detection

**DeepSeek**:
- OpenRouter (starts with `sk-or-`): Uses `https://openrouter.ai/api/v1`
- Direct DeepSeek: Uses `https://api.deepseek.com`

---

## 9. KEY FEATURES & CAPABILITIES

### RAG System
- Multi-document semantic search (TF-IDF)
- Context-aware generation (DeepSeek, OpenAI, Anthropic)
- Async/concurrent processing
- Pluggable LLM providers

### Evaluation Framework
- Two-tier evaluation (rule-based + LLM-judge)
- Semantic similarity understanding
- Confidence scoring
- Detailed reasoning explanations
- Batch evaluation with comprehensive metrics

### Agent System
- Dual-agent architecture (White/Green)
- A2A protocol compliance
- Stateful conversation management
- Integration with Tau-Bench benchmarks

### Data Management
- PDF parsing with semantic chunking
- Configurable chunk sizes and overlap
- Efficient vector database serialization
- Ground truth dataset management

---

## 10. CURRENT LIMITATIONS & NOTES

1. **API Dependencies**: Requires valid API keys for LLM-based evaluation
2. **Vector DB**: Currently uses TF-IDF (not neural embeddings)
3. **White Agent**: Currently a stub; designed for future expansion
4. **Agentify Integration**: Tau-Bench specific; may need adaptation for other benchmarks
5. **Data Size**: Limited to 183 chunks (2 documents); easily expandable
6. **Query Set**: 7+ predefined queries; easily extensible

---

## 11. EXTENSION POINTS

### Adding New Queries
Edit `data/predefined_queries.json`:
```json
{
  "id": 8,
  "query": "Your question here?",
  "ground_truth": "Expected answer"
}
```

### Adding New Documents
1. Place PDFs in `docs/`
2. Run: `python scripts/parse_pdfs_to_json.py`
3. Rebuild DB: `python simple_vector_db.py --json_file data/safety_datasets.json --save`

### Adding LLM Provider
Edit `utils/llm_client.py`:
```python
# Add new provider class
class CustomLLMClient(BaseLLMClient):
    # Implement generate_response() and generate_chat_response()
    pass

# Register in LLMClient.__init__()
```

### Custom Evaluation Metrics
Extend `green_agent/evaluation.py`:
```python
class CustomEvaluator:
    def evaluate(self, response, ground_truth):
        # Custom logic
        pass
```

---

## 12. SUMMARY TABLE

| Component | Purpose | Type | Key Files |
|-----------|---------|------|-----------|
| RAG Pipeline | Query→Answer | Core | safety_datasets_rag.py |
| Vector DB | Semantic Search | Core | simple_vector_db.py |
| Evaluation | Response Validation | Core | green_agent/evaluation.py |
| Green Agent | Query Interface | Agent | green_agent/agent.py |
| White Agent | Executor Stub | Agent | white_agent/agent.py |
| LLM Client | Multi-provider LLM | Utility | utils/llm_client.py |
| A2A Framework | Agent Communication | System | agentify-example-tau-bench/ |
| PDF Parser | Document Processing | Utility | scripts/parse_pdfs_to_json.py |
| Data | Queries + Chunks | Assets | data/ |

---

## 13. QUICK START

### Minimal Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env with API key
cp env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# 3. Run predefined queries
python green_agent/agent.py --all
```

### Full Setup with Custom Data
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place PDFs in docs/
cp your_documents.pdf docs/

# 3. Parse PDFs
python scripts/parse_pdfs_to_json.py

# 4. Build vector database
python simple_vector_db.py --json_file data/safety_datasets.json --save

# 5. Add queries to data/predefined_queries.json

# 6. Run evaluation
python green_agent/agent.py --all --use_llm_judge
```

### A2A Agent Evaluation
```bash
cd agentify-example-tau-bench
uv sync
OPENAI_API_KEY=... uv run python main.py launch
```

---

## 14. ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                    AIPolicyBench System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        User Interface Layer                             │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  Interactive CLI  │  Batch Mode  │  API Endpoints      │  │
│  └────────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        Agent & Evaluation Layer                         │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │ Green Agent (Query Interface)                           │  │
│  │   ├─ Rule-Based Evaluator                              │  │
│  │   └─ LLM-Judge Evaluator                               │  │
│  │ White Agent (Task Executor)                            │  │
│  └────────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        RAG Pipeline Layer                               │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  1. Retrieval   │  2. Augmentation  │  3. Generation   │  │
│  │  (TF-IDF)       │  (Context Prep)   │  (LLM)           │  │
│  └────────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ↓                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        Data & Storage Layer                             │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │ Vector DB          │ Safety Datasets    │ Queries       │  │
│  │ (183 chunks)       │ (183 chunks)       │ (7+ Q/A)      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        LLM Integration Layer                            │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │ DeepSeek  │  OpenAI  │  Anthropic  │  Local Models    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘

A2A Framework (Separate):
┌─────────────────────────────────────────────────────────────┐
│              Agentify Tau-Bench Integration                  │
├─────────────────────────────────────────────────────────────┤
│  Green Agent Server ↔ A2A Protocol ↔ White Agent Server   │
│  (Evaluator)                            (Task Executor)     │
│                                                               │
│  ├─ Agent Card Registry                                     │
│  ├─ Message Routing                                         │
│  └─ Task Management                                         │
└─────────────────────────────────────────────────────────────┘
```

---

End of AIPolicyBench Comprehensive Overview
