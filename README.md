# 🎯 AI Policy Agent Benchmark - Safety Datasets RAG System

A complete **Retrieval-Augmented Generation (RAG)** system for AI policy research, built to answer questions about safety evaluation datasets using natural language responses.

## 🚨 **IMPORTANT: API Key Issue**

**Current Status:** The DeepSeek API keys are currently showing as invalid (401 authentication errors). This appears to be a recent issue with DeepSeek's authentication system.

**Solution:** Please use a **new DeepSeek API key from OpenRouter** or generate a fresh key from your DeepSeek account.

**To get a working API key:**
1. Visit [OpenRouter](https://openrouter.ai/) and get a DeepSeek API key
2. Or generate a new key from [DeepSeek Platform](https://platform.deepseek.com)
3. Update the `env.example` file with your new key

---

## 🏗️ **System Architecture**

### **Core Components:**

1. **`safety_datasets_rag.py`** - Core RAG pipeline (Steps 1-4)
2. **`query_interface.py`** - Clean interface for natural language responses  
3. **`simple_vector_db.py`** - TF-IDF vector database implementation
4. **`data/safety_datasets.csv`** - 149 safety evaluation datasets
5. **`vector_db/`** - Serialized vector database

### **RAG Pipeline (4 Steps):**

```
📊 Step 1: RETRIEVAL    → Find relevant datasets from 149 options
🔗 Step 2: AUGMENTATION → Prepare rich context with metadata  
🤖 Step 3: GENERATION   → DeepSeek LLM creates natural response
💬 Step 4: RESPONSE     → Return actionable policy guidance
```

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Set Up API Key**
```bash
# Copy and edit the environment file
cp env.example .env
# Add your DeepSeek API key to .env
# The system will automatically load API keys from .env
```

### **3. Build Vector Database** (if needed)
```bash
python simple_vector_db.py --csv_file data/safety_datasets.csv --save
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
AIPolicyAgentBench/
├── 🎯 query_interface.py          # Predefined query evaluation interface
├── 🧠 safety_datasets_rag.py      # Core RAG pipeline (Steps 1-4)
├── 🔍 simple_vector_db.py         # TF-IDF vector database
├── 📊 evaluation.py               # Rule-based evaluation system
├── 📊 data/
│   ├── safety_datasets.csv        # 149 safety evaluation datasets
│   └── predefined_queries.json    # Predefined queries with ground truth
├── 💾 vector_db/
│   └── safety_datasets_tfidf_db.pkl  # Serialized vector database
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
- Predefined queries with ground truth answers
- Rule-based evaluation (exact match, substring, set matching)
- Batch evaluation with accuracy statistics
- Individual query evaluation
- Interactive query selection
- Detailed evaluation metrics and reporting

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
  --queries_file TEXT   Path to predefined queries JSON file (default: data/predefined_queries.json)
  --vector_db TEXT      Path to vector database file (default: ./vector_db/safety_datasets_tfidf_db.pkl)
  --query_id INTEGER    Evaluate specific query by ID
  --all                 Evaluate all queries
  --top_k INTEGER       Number of datasets to retrieve (default: 5)
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
- **Source:** 149 safety evaluation datasets
- **Fields:** Name, purpose, type, languages, entries, publication, URLs, license
- **Preprocessing:** Text normalization and metadata extraction

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
# Rebuild the vector database
python simple_vector_db.py --csv_file data/safety_datasets.csv --save
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