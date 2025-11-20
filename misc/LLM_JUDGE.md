### LLM Judge (Concise)

What it does
- Classifies responses: correct, miss (uncertain), hallucination (wrong)
- Returns: classification, confidence (0..1), reasoning

When to use
- For semantic correctness beyond substring matches

Run with judge
```bash
python main.py launch \ 
  --vector-db ./vector_db/safety_datasets_tfidf_db.pkl \ 
  --white-model deepseek-chat --llm-judge
```

API keys
- Put provider keys in `.env` (e.g., `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`)
- OpenRouter keys supported (`sk-or-*`)

Metrics
- Correct% = correct/total ×100
- Miss% = misses/total ×100
- Hallucination% = hallucinations/total ×100
- Factuality% = Correct% - Hallucination% + 0.5×Miss%

Outputs
- Per‑query JSON (judgment + reasoning)
- `summary.json` (counts + rates)
- `statistics.txt` (human‑readable)

Tips
- Prefer lower hallucination even if miss increases slightly
- Use rule‑based for fast loops; judge for final scoring

