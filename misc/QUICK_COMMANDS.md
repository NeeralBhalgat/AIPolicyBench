### TL;DR: Essential Commands (Concise)

- Run white agent (serve RAG answers)
```bash
cd /Users/BerkeleyClasses/Junior/CS194/AIPolicyBench-1
source .venv/bin/activate
python main.py white --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --model deepseek-chat
```

- One-shot evaluation (starts green + white)
```bash
python main.py launch \
  --queries-file data/predefined_queries.json \
  --vector-db ./vector_db/safety_datasets_tfidf_db.pkl \
  --white-model deepseek-chat
```

- Quick tests
```bash
python green_agent/evaluation.py                # Rule-based demo
python green_agent/agent.py --query_id 1        # Single query
python green_agent/agent.py --all               # All queries
python test_a2a_imports.py                      # A2A imports
python test_llm_judge.py                        # Judge prompts
```

- Build/update vector DB
```bash
python simple_vector_db.py --json_file data/safety_datasets.json --save
# Output: ./vector_db/safety_datasets_tfidf_db.pkl
```

- Reproduce results (variants)
```bash
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model mistralai/mistral-7b-instruct
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model openai/gpt-5.1
```

- Optional: LLM-as-a-judge
```bash
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat --llm-judge
```

- Outputs
- results/<model>/: summary.json, statistics.txt, per‑query JSON

