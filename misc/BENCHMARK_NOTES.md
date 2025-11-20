### Benchmark Notes (Concise)

Implementation Summary
- White agent wraps RAG; green agent coordinates and evaluates (A2A)
- Launcher starts both agents and sends evaluation task
- Vector DB (TF‑IDF) enables fast retrieval

Run
```bash
# Build DB
python simple_vector_db.py --json_file data/safety_datasets.json --save

# Evaluate (example model)
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat
```

Judge Option
```bash
python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat --llm-judge
```

Model Config (White Agent)
- Default: `deepseek-chat`
- Alternatives: `mistralai/mistral-7b-instruct`, `openai/gpt-5.1` (OpenRouter)\n- Set via `--white-model` in launch or `--model` in white

API Keys
- `.env`: `DEEPSEEK_API_KEY` (OpenRouter `sk-or-*` supported), `OPENAI_API_KEY` (if used)\n- Ensure keys before running judge or generation

Updates (Highlights)
- Results auto-saved under `results/<model>/` (`summary.json`, `statistics.txt`)\n- Hyphenated Typer flags (`--vector-db`, `--white-model`) for consistency

Metrics
- Correct%, Miss%, Hallucination%, Factuality% = Correct% - Hallucination% + 0.5×Miss%

Tips\n- Keep hallucinations low; uncertainty (miss) is safer than wrong\n- Use judge for final reporting; rule-based for fast iterations

