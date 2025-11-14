# AIPolicyBench — README 

This README documents the current project layout and the exact commands to:
1. Run white agents (generate responses for predefined queries).
2. Run the green agent (LLM-as-a-judge) to evaluate white agents.
3. Run tests / example test cases that exercise the green agent evaluator.

This repo contains:
- Python evaluation & RAG code: query_interface.py, safety_datasets_rag.py, simple_vector_db.py, evaluation.py
- Web leaderboard (React + Vite): agent-leaderboard-web/
- Data and scripts: data/, scripts/, vector_db/

Prerequisites
- Python 3.9+ and pip
- A valid LLM API key in `.env` (DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
- Node.js & npm (if you want the web leaderboard)

Quick setup (macOS terminal)
```bash
# Python deps
cd /Users/isabelle/Desktop/AIPolicyBench
pip install -r requirements.txt

# Copy .env and add API key(s)
cp env.example .env
# edit .env to include DEEPSEEK_API_KEY or other provider key

# (Optional) Web UI deps
cd agent-leaderboard-web
npm install
```

1) Running white agents (generate responses)
- Purpose: run a white agent (an LLM) to answer the predefined queries (generation step).
- Use `query_interface.py`. Choose provider with `--llm_provider` (deepseek, openai, anthropic).

Examples:
```bash
# Run all predefined queries with DeepSeek as the white agent
cd /Users/isabelle/Desktop/AIPolicyBench
python query_interface.py --all --llm_provider deepseek

# Run a single query (id=1) with OpenAI as the white agent
python query_interface.py --query_id 1 --llm_provider openai

# Save stdout to file (capture white-agent outputs)
python query_interface.py --all --llm_provider deepseek > results/white_deepseek_all.txt
```

Notes:
- The generation step uses the RAG pipeline (retrieval + augmentation) internally (safety_datasets_rag.py).
- You can change retrieval top-k with `--top_k N` when calling `query_interface.py`.

2) Running the green agent (LLM-as-a-judge) to evaluate white agents
- Purpose: use a trusted LLM ("green agent") to semantically judge white-agent responses.
- Two options: run the judge via `query_interface.py` (integrated flow) or use `evaluation.py` directly.

Examples (integrated):
```bash
# Place PDF documents in ./docs directory
# Parse PDFs and generate JSON with intelligent chunking
python scripts/parse_pdfs_to_json.py --chunk_size 1500 --overlap 300

# Build vector database from parsed JSON
python simple_vector_db.py --json_file data/safety_datasets.json --save
```

#### **Option B: Use Existing Data**
```bash
# If you already have safety_datasets.json, just build the database
python simple_vector_db.py --json_file data/safety_datasets.json --save
```

### **4. Run Predefined Query Evaluation**

#### **Interactive Mode:**
```bash
python query_interface.py
```

#### **Evaluate Specific Query by ID:**
```bash
python query_interface.py --query_id 1
```

#### **Evaluate All Predefined Queries:**
```bash
python query_interface.py --all
```

**Note:** The system uses predefined queries from `data/predefined_queries.json` with ground truth answers for automated evaluation. API keys are automatically loaded from `.env` file.

---

## 💬 **Predefined Queries with Ground Truth Evaluation**

The system includes predefined queries with known ground truth answers for automated evaluation. Queries are stored in `data/predefined_queries.json`.

### **Example 1: Query by ID**
```bash
python query_interface.py --query_id 1
```

**Expected Output:**
```
================================================================================
Query ID: 1
================================================================================

📝 Query: Who is the admin work lead of AIPolicyBench teams?
--------------------------------------------------------------------------------

💬 Model Response:
Isabella

✓ Ground Truth: Isabella

✅ Evaluation: CORRECT
📊 Method: Rule-based
📚 Retrieved Datasets: 5
================================================================================
```

### **Example 2: Evaluate All Queries**
```bash
python query_interface.py --all
```

**Expected Output:**
```
================================================================================
🎯 AIPOLICYBENCH EVALUATION RESULTS
================================================================================
[Individual query results displayed here]

================================================================================
📊 OVERALL STATISTICS
================================================================================
Total Queries: 3
Correct: 3
Incorrect: 0
Accuracy: 100.00%
Evaluation Method: Rule-based
================================================================================
```

### **Example 3: Interactive Mode**
```bash
python query_interface.py

# Commands:
#   1-3: Evaluate specific query
#   all: Evaluate all queries
#   list: Show all queries
#   quit/exit: Exit
```

---

## 📁 **File Structure**

```
AIPolicyBench/
├── 🎯 query_interface.py          # Predefined query evaluation interface
├── 🧠 safety_datasets_rag.py      # Core RAG pipeline (Steps 1-4)
├── 🔍 simple_vector_db.py         # TF-IDF vector database
├── 📊 evaluation.py               # Evaluation system (Rule-based + LLM-as-a-judge)
│                                  # - RuleBasedEvaluator: Fast pattern matching
│                                  # - LLMJudgeEvaluator: Semantic evaluation with Green Agent
│                                  # - Metrics: Correct/Hallucination/Miss/Factuality rates
├── 🔧 convert_csv_to_json.py      # CSV to JSON converter
├── 📂 scripts/
│   ├── parse_pdfs_to_json.py      # PDF parsing and chunking utility
│   └── README.md                  # Scripts documentation
├── 📊 data/
│   ├── safety_datasets.json       # Processed document chunks (main dataset)
│   ├── safety_datasets.csv        # Legacy CSV format (for reference)
│   └── predefined_queries.json    # Predefined queries with ground truth
├── 📄 docs/
│   ├── 1 Americas-AI-Action-Plan.pdf
│   └── 5 Artificial Intelligence Risk Management Framework (AI RMF 1.0).pdf
├── 💾 vector_db/
│   └── safety_datasets_tfidf_db.pkl  # Serialized vector database
├── 🛠️ utils/
│   └── llm_client.py              # LLM client implementations (OpenAI, DeepSeek, Anthropic)
├── ⚙️ .env                        # Environment variables (API keys)
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # This file
```

---

## 🔧 **System Features**

### **✅ What Works Perfectly:**
- **Vector Database:** 149 datasets indexed with TF-IDF
- **Retrieval:** Semantic search finds relevant datasets
- **Augmentation:** Rich context preparation with metadata
- **Interface:** Clean CLI and interactive modes
- **Evaluation:** Rule-based evaluation with ground truth answers
- **Error Handling:** Graceful fallbacks and logging
- **API Integration:** Automatic API key loading from .env

### **🎯 Evaluation System Features:**
- **Two Evaluation Methods:**
  - **Rule-based**: Fast substring matching with uncertainty detection
  - **LLM-as-a-Judge**: Advanced semantic evaluation by a trusted LLM (Green Agent)
- **Comprehensive Metrics:**
  - Correct Rate: Questions answered correctly
  - Hallucination Rate: Factually incorrect responses
  - Miss Rate: Uncertainty expressions ("I don't know")
  - Factuality Rate: `Correct - Hallucination + c × Miss` (rewards uncertainty over wrong answers)
- Predefined queries with ground truth answers
- Batch evaluation with detailed statistics
- Individual query evaluation with confidence scores and reasoning
- Interactive query selection
- Support for multiple LLM providers (DeepSeek, OpenAI, Anthropic)

---

## 📊 **Evaluation Methods**

The system supports two evaluation approaches following a Question-Answering (QA) paradigm:

### **1. Rule-Based Evaluation (Fast, No API Required)**

Uses pattern matching and uncertainty detection to classify responses into three categories:

- **✅ Correct**: Response contains the ground truth answer
- **⚠️ Miss**: Response expresses uncertainty (e.g., "I don't know", "I'm not sure")
- **❌ Hallucination**: Response attempts to answer but is factually incorrect

**When to use:** Fast evaluation, no API costs, good for exact matches and simple validation.

```bash
# Use rule-based evaluation (default)
python query_interface.py --all
```

### **2. LLM-as-a-Judge Evaluation (Advanced, Requires API)**

Uses a trusted LLM (Green Agent) to evaluate White Agent responses with semantic understanding:

- Understands semantic equivalence (e.g., "FDA" vs "Food and Drug Administration")
- Provides confidence scores (0-1) for each judgment
- Includes detailed reasoning for each classification
- Better handles paraphrasing and complex answers

**When to use:** Need semantic understanding, evaluating complex answers, want detailed reasoning.

```bash
# Use LLM-as-a-judge with DeepSeek (default)
python query_interface.py --all --use_llm_judge

# Use with OpenAI
python query_interface.py --all --use_llm_judge --llm_provider openai

# Use with Anthropic Claude
python query_interface.py --all --use_llm_judge --llm_provider anthropic
```

### **Evaluation Metrics Explained**

The system calculates four key metrics:

1. **Correct Rate** = (Correct / Total) × 100%
   - Higher is better
   - Measures accuracy of correct answers

2. **Hallucination Rate** = (Hallucinations / Total) × 100%
   - Lower is better
   - Measures factually incorrect responses

3. **Miss Rate** = (Misses / Total) × 100%
   - Neutral metric
   - Measures when agent admits uncertainty

4. **Factuality Rate** = Correct Rate - Hallucination Rate + (c × Miss Rate)
   - Higher is better
   - Default: c = 0.5 (miss weight)
   - Rewards uncertainty over wrong answers
   - Best overall measure of agent reliability

**Example:**
- 6 questions total
- 3 correct (50%)
- 1 hallucination (16.67%)
- 2 misses (33.33%)
- **Factuality Rate** = 50% - 16.67% + (0.5 × 33.33%) = **50%**

---

## 🧪 **Evaluation Examples**

### **Example 1: Rule-Based Evaluation**

```bash
python query_interface.py --query_id 1
```

**Output:**
```
================================================================================
Query ID: 1
================================================================================

📝 Query: Who is the admin work lead of AIPolicyBench teams?
--------------------------------------------------------------------------------

💬 Model Response:
The admin work lead of AIPolicyBench teams is Isabella.

✓ Ground Truth: Isabella

✅ Evaluation: CORRECT
📊 Method: Rule-based
📚 Retrieved Datasets: 5
================================================================================
```

### **Example 2: LLM-as-a-Judge Evaluation**

```bash
python query_interface.py --query_id 1 --use_llm_judge
```

**Output:**
```
================================================================================
Query ID: 1
================================================================================

📝 Query: Who is the admin work lead of AIPolicyBench teams?
--------------------------------------------------------------------------------

💬 Model Response:
The admin work lead of AIPolicyBench teams is Isabella.

✓ Ground Truth: Isabella

✅ Evaluation: CORRECT
📊 Method: LLM-as-a-judge
🎯 Confidence: 0.95
💭 Reasoning: The response correctly identifies Isabella as the admin work lead, matching the ground truth exactly.
📚 Retrieved Datasets: 5
================================================================================
```

### **Example 3: Batch Evaluation with All Metrics**

```bash
python query_interface.py --all --use_llm_judge
```

**Output:**
```
================================================================================
📊 OVERALL STATISTICS
================================================================================
Total Queries: 6
Correct: 3
Hallucination: 1
Miss: 2

Correct Rate: 50.00%
Hallucination Rate: 16.67%
Miss Rate: 33.33%
Factuality Rate: 50.00%

Evaluation Method: LLM-as-a-judge
LLM Provider: deepseek
================================================================================
```

### **Example 4: Testing Evaluation Module Directly**

```bash
# Run the evaluation module demo
conda run -n aipolicy python evaluation.py
```

This will test both correct answers, misses, and hallucinations with sample data.

---

## 🤖 **Using LLM-as-a-Judge in Your Code**

### **Quick Start Example**

```python
import asyncio
from evaluation import LLMJudgeEvaluator

async def evaluate_responses():
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        provider="deepseek",  # or "openai", "anthropic"
        temperature=0.0       # Deterministic evaluation
    )

    # Evaluate a single response
    result = await evaluator.evaluate(
        response="The FDA regulates AI in medical devices.",
        ground_truth="FDA",
        question="Which agency regulates AI in medical devices?"
    )

    print(f"Classification: {result['result']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reasoning: {result['reasoning']}")

asyncio.run(evaluate_responses())
```

### **Batch Evaluation Example**

```python
import asyncio
from evaluation import LLMJudgeEvaluator, RuleBasedEvaluator

async def compare_evaluators():
    questions = [
        "Who leads the evaluation part?",
        "How many team members are there?",
        "What is the project focus?"
    ]

    responses = [
        "Charles leads the evaluation.",
        "I'm not sure about the team size.",
        "The project focuses on policy."  # Hallucination if wrong
    ]

    ground_truths = ["Charles", "4", "AI safety"]

    # Rule-based evaluation
    rule_eval = RuleBasedEvaluator()
    rule_result = rule_eval.evaluate_batch(responses, ground_truths)

    # LLM-as-a-judge evaluation
    llm_eval = LLMJudgeEvaluator(provider="deepseek")
    llm_result = await llm_eval.evaluate_batch(
        responses=responses,
        ground_truths=ground_truths,
        questions=questions
    )

    print("Rule-based Factuality:", rule_result['statistics']['factuality_rate'])
    print("LLM Judge Factuality:", llm_result['statistics']['factuality_rate'])

asyncio.run(compare_evaluators())
```

### **Custom Miss Weight**

```python
# Adjust how much we reward uncertainty
result = evaluator.evaluate_batch(
    responses=responses,
    ground_truths=ground_truths,
    miss_weight=0.7  # Higher reward for admitting uncertainty
)
```

### **Evaluation Method Selection**

Use **Rule-Based** when:
- ✅ You need fast, cheap evaluation
- ✅ Ground truth is a simple string or number
- ✅ Exact/substring matching is sufficient
- ✅ No API costs acceptable

Use **LLM-as-a-Judge** when:
- ✅ Responses require semantic understanding
- ✅ Need confidence scores and reasoning
- ✅ Paraphrasing and variations are expected
- ✅ Want to understand WHY something was classified
- ⚠️ API costs are acceptable

---

## 📄 **PDF Document Processing**

The system includes a powerful PDF parsing and chunking utility to process policy documents.

### **How It Works:**

1. **Text Extraction**: Extracts text from PDF files using PyPDF2
2. **Intelligent Chunking**: Splits documents into semantic chunks
   - Preserves paragraph boundaries
   - Splits long paragraphs at sentence boundaries
   - Configurable chunk size with overlap for context
3. **Clean Output**: Generates JSON with only `id` and `text` fields
4. **Source Tracking**: Each chunk includes source PDF filename

### **Basic Usage:**

```bash
# Place PDF files in ./docs directory
cp your_policy_document.pdf docs/

# Parse all PDFs with default settings (1500 chars, 300 overlap)
python scripts/parse_pdfs_to_json.py

# Rebuild vector database
python simple_vector_db.py --json_file data/safety_datasets.json --save
```

### **Custom Chunk Sizes:**

```bash
# Smaller chunks for precise retrieval
python scripts/parse_pdfs_to_json.py --chunk_size 1000 --overlap 200

# Larger chunks for more context
python scripts/parse_pdfs_to_json.py --chunk_size 2000 --overlap 400
```

### **Advanced Options:**

```bash
python scripts/parse_pdfs_to_json.py \
  --docs_dir ./documents \
  --output ./output/custom_data.json \
  --chunk_size 1500 \
  --overlap 300
```

### **Chunk Size Recommendations:**

- **Small (500-1000 chars)**: Better precision, more chunks, good for Q&A
- **Medium (1000-2000 chars)**: Balanced, recommended for policy documents
- **Large (2000-3000 chars)**: More context, fewer chunks, good for summaries

### **Output Format:**

The script generates `data/safety_datasets.json`:

```json
[
  {
    "id": "Americas-AI-Action-Plan_chunk_0",
    "text": "Source: Americas-AI-Action-Plan.pdf\n\nChunk content..."
  }
]
```

### **Current Dataset:**

- **15 PDF documents** processed
- **200 Questions** produced with Answers and Quoted Reference

---

## 🛠️ **Advanced Usage**

### **Interactive Mode Commands:**
- `1-N` - Evaluate specific query by ID
- `all` - Evaluate all predefined queries
- `list` - Show all available queries with ground truth
- `quit/exit` - Exit the system

### **Command Line Options:**
```bash
python query_interface.py [OPTIONS]

Options:
  --queries_file TEXT      Path to predefined queries JSON file
                          (default: data/predefined_queries.json)
  --vector_db TEXT        Path to vector database file
                          (default: ./vector_db/safety_datasets_tfidf_db.pkl)
  --query_id INTEGER      Evaluate specific query by ID
  --all                   Evaluate all queries
  --top_k INTEGER         Number of datasets to retrieve (default: 5)
  --use_llm_judge         Use LLM-as-a-judge evaluation instead of rule-based
  --llm_provider TEXT     LLM provider for judge evaluation
                          Choices: deepseek, openai, anthropic (default: deepseek)
```

**Examples:**
```bash
# Rule-based evaluation (default, fast, no API cost)
python query_interface.py --all

# LLM-as-a-judge with DeepSeek
python query_interface.py --all --use_llm_judge

# LLM-as-a-judge with OpenAI GPT-4
python query_interface.py --all --use_llm_judge --llm_provider openai

# Single query with LLM judge
python query_interface.py --query_id 1 --use_llm_judge
```

### **Adding Custom Queries:**
Edit `data/predefined_queries.json` to add new queries:
```json
{
  "queries": [
    {
      "id": 1,
      "query": "Your question here?",
      "ground_truth": "Expected answer"
    }
  ]
}
```

---

## 🔍 **Technical Details**

### **Vector Database:**
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features:** 5000 max features, (1,2) n-gram range
- **Similarity:** Cosine similarity for ranking
- **Storage:** Pickle serialization for fast loading

### **LLM Integration:**
- **Provider:** DeepSeek via OpenRouter (OpenAI-compatible API)
- **Model:** `deepseek/deepseek-chat` (OpenRouter) or `deepseek-chat` (Direct)
- **Context:** Up to 1500 tokens
- **Temperature:** 0.3 for focused, accurate responses
- **API Key:** Automatically loaded from `.env` file (DEEPSEEK_API_KEY)

### **Data Processing:**
- **Source:** PDF policy documents in `./docs` directory
- **Extraction:** PyPDF2 for text extraction from PDFs
- **Chunking Strategy:**
  - Semantic chunking by paragraphs
  - Sentence-level splitting for long paragraphs
  - Default: 1500 chars with 300 char overlap
  - Maintains context across chunks
- **Format:** Simple JSON with `id` and `text` fields
- **Current Dataset:** 183 chunks from 2 PDF documents

---

## 🚨 **Troubleshooting**

### **API Key Issues:**
1. Ensure `.env` file exists with valid `DEEPSEEK_API_KEY`
2. For OpenRouter keys (starting with `sk-or-`), the system auto-detects and uses OpenRouter endpoint
3. Test your API key:
```bash
# For OpenRouter
curl -X POST "https://openrouter.ai/api/v1/chat/completions" \
  -H "Authorization: Bearer sk-or-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": "Hello"}]}'
```

### **Predefined Queries Not Found:**
```bash
# Ensure predefined_queries.json is in the data directory
ls data/predefined_queries.json

# If missing, create it with your queries
```

### **Vector Database Missing:**
```bash
# Rebuild the vector database from JSON
python simple_vector_db.py --json_file data/safety_datasets.json --save
```

### **PDF Parsing Issues:**
```bash
# Install PDF parsing library
pip install PyPDF2

# Or use alternative
pip install pdfplumber

# If no text extracted (image-based PDFs), you need OCR
pip install pytesseract pdf2image
```

### **Chunks Too Large/Small:**
```bash
# Adjust chunk size to your needs
python scripts/parse_pdfs_to_json.py --chunk_size 1000 --overlap 200

# Check statistics in output to verify chunk sizes
```

### **Import Errors:**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

---

## 🤝 **Contributing**

This system is designed for AI policy research and can be extended for:
- Additional dataset sources
- Different LLM providers
- Web interface development
- Advanced retrieval methods
- Policy-specific fine-tuning

---

## 📄 **License**

This project is for research and educational purposes. Please ensure compliance with dataset licenses and API terms of service.

---

## 🆘 **Support**

If you encounter issues:
1. Check the API key is valid in `.env` file (DEEPSEEK_API_KEY)
2. Verify all dependencies are installed (`pip install -r requirements.txt`)
3. Ensure the vector database exists at `./vector_db/safety_datasets_tfidf_db.pkl`
4. Verify predefined queries file exists at `data/predefined_queries.json`
5. Check API key has sufficient credits

**For API key issues:** Get a fresh DeepSeek key from [OpenRouter](https://openrouter.ai/) or [DeepSeek Platform](https://platform.deepseek.com)