<<<<<<< HEAD
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
```

### **3. Build Vector Database** (if needed)
```bash
python simple_vector_db.py --csv_file data/safety_datasets.csv --save
```

### **4. Ask Questions!**

#### **Interactive Mode:**
```bash
python query_interface.py --api_key YOUR_DEEPSEEK_KEY
```

#### **Single Question:**
```bash
python query_interface.py --question "What datasets evaluate AI bias?" --api_key YOUR_KEY
```

#### **Retrieval Only (no LLM):**
```bash
python query_interface.py --question "What datasets evaluate AI safety?" --no-llm
```

---

## 💬 **Sample Questions & Expected Outputs**

### **Example 1: AI Bias Evaluation**
```bash
python query_interface.py --question "What datasets are available for evaluating AI bias and fairness in decision-making systems?" --api_key YOUR_KEY
```

**Expected Output:**
```
🎯 Question: What datasets are available for evaluating AI bias and fairness in decision-making systems?
================================================================================

📊 Found 5 relevant datasets:
----------------------------------------
1. DeMET (Score: 0.277)
   Purpose: Demographic bias evaluation
   Publication: EMNLP 2024 (Findings)

2. GenMO (Score: 0.207)  
   Purpose: Gender bias assessment
   Publication: EMNLP 2024 (Findings)

💬 Answer:
----------------------------------------
Based on the safety evaluation datasets I found, there are several excellent options for evaluating AI bias and fairness in decision-making systems:

The **DeMET dataset** (EMNLP 2024) is particularly valuable for assessing demographic bias in AI systems. This dataset can help identify how AI models perpetuate or challenge biases in decision-making scenarios across different demographic groups.

The **GenMO dataset** provides comprehensive gender bias testing capabilities, offering a robust framework for evaluating fairness in AI decision-making algorithms, especially in high-stakes applications like hiring, lending, or healthcare.

For implementation, I recommend starting with DeMET for general demographic bias testing, then incorporating GenMO for gender-specific assessments. These datasets provide the foundation for building trustworthy AI decision-making systems that comply with fairness regulations.

✅ Generated using DeepSeek LLM with 5 datasets
================================================================================
```

### **Example 2: Healthcare AI Safety**
```bash
python query_interface.py --question "What datasets evaluate AI safety in healthcare applications?" --api_key YOUR_KEY
```

### **Example 3: Multilingual Safety**
```bash
python query_interface.py --question "Which datasets assess AI safety in Chinese language models?" --api_key YOUR_KEY
```

### **Example 4: Policy Compliance**
```bash
python query_interface.py --question "What datasets help evaluate AI compliance with government regulations?" --api_key YOUR_KEY
```

---

## 📁 **File Structure**

```
AIPolicyAgentBench/
├── 🎯 query_interface.py          # Main interface (natural language responses)
├── 🧠 safety_datasets_rag.py      # Core RAG pipeline (Steps 1-4)
├── 🔍 simple_vector_db.py         # TF-IDF vector database
├── 📊 data/
│   └── safety_datasets.csv        # 149 safety evaluation datasets
├── 💾 vector_db/
│   └── safety_datasets_tfidf_db.pkl  # Serialized vector database
├── ⚙️ env.example                 # Environment variables template
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
- **Error Handling:** Graceful fallbacks and logging

### **⚠️ Current Issue:**
- **LLM Generation:** DeepSeek API authentication failing (401 errors)
- **Workaround:** Use `--no-llm` flag for retrieval-only mode

### **🎯 Natural Language Responses Include:**
- Direct answers to policy questions
- Specific dataset recommendations with scores
- Implementation guidance and best practices
- Key considerations and limitations
- Relevant resources and publication links
- Actionable next steps for policy makers

---

## 🛠️ **Advanced Usage**

### **Interactive Mode Commands:**
- `help` - Show example questions
- `status` - Check system status
- `no-llm` - Toggle LLM generation on/off
- `quit` - Exit the system

### **Command Line Options:**
```bash
python query_interface.py [OPTIONS]

Options:
  --api_key TEXT        DeepSeek API key for LLM generation
  --question TEXT       Single question (non-interactive mode)
  --top_k INTEGER       Number of datasets to retrieve (default: 5)
  --no-llm             Disable LLM generation (retrieval only)
  --vector_db TEXT      Path to vector database file
```

### **System Status Check:**
```bash
python query_interface.py
# Then type: status
```

---

## 🔍 **Technical Details**

### **Vector Database:**
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features:** 5000 max features, (1,2) n-gram range
- **Similarity:** Cosine similarity for ranking
- **Storage:** Pickle serialization for fast loading

### **LLM Integration:**
- **Provider:** DeepSeek (OpenAI-compatible API)
- **Model:** `deepseek-chat`
- **Context:** Up to 1500 tokens
- **Temperature:** 0.7 for balanced creativity/accuracy

### **Data Processing:**
- **Source:** 149 safety evaluation datasets
- **Fields:** Name, purpose, type, languages, entries, publication, URLs, license
- **Preprocessing:** Text normalization and metadata extraction

---

## 🚨 **Troubleshooting**

### **API Key Issues:**
```bash
# Test your API key directly
curl -X POST "https://api.deepseek.com/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello"}]}'
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
1. Check the API key is valid and has credits
2. Verify all dependencies are installed
3. Ensure the vector database exists
4. Try retrieval-only mode with `--no-llm`

**For API key issues:** Get a fresh DeepSeek key from OpenRouter or DeepSeek Platform.
=======
# AIPolicyBench
Complete RAG system for AI policy research with safety datasets
>>>>>>> d22caee0f3038c151f68cd8b5bcdfd5910bc2650
