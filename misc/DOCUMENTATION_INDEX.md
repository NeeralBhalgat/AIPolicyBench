<<<<<<< HEAD
# AIPolicyBench - Documentation Index

Welcome! This document helps you navigate the comprehensive documentation for the AIPolicyBench codebase.

## Available Documentation

### 1. CODEBASE_OVERVIEW.md (763 lines)
**Comprehensive architectural and technical documentation**

Covers:
- Project summary and core architecture
- Detailed component descriptions (RAG, evaluation, agents)
- A2A (Agent-to-Agent) system design
- Data structures and formats
- Execution flows and workflows
- CLI interfaces and command examples
- Dependencies and configuration
- Extension points and customization
- Architecture diagrams

**Best for**: Understanding the overall system design, deep technical details, how components interact

### 2. QUICK_REFERENCE.md (278 lines)
**Fast lookup guide with common operations**

Covers:
- Project structure overview
- Component-to-file mapping table
- Quick start (4 steps)
- Core workflows (RAG pipeline, evaluation, A2A)
- Key methods and APIs
- Configuration settings
- Common commands
- Troubleshooting table
- Performance notes
- Key insights

**Best for**: Quick lookups, API reference, getting started fast, troubleshooting

### 3. FILE_STRUCTURE.md (371 lines)
**Complete directory tree and file-by-file breakdown**

Covers:
- Full directory tree with descriptions
- Detailed file descriptions (purpose, size, classes, methods)
- Data files specification
- A2A framework components
- Configuration files
- Key absolute file paths
- Import structure
- File statistics
- Git repository info

**Best for**: Finding specific files, understanding file purposes, import statements, directory organization

## Navigation Guide

### I want to understand...

**The overall architecture:**
- Start with: CODEBASE_OVERVIEW.md (Section 1: Core Architecture)
- Then read: FILE_STRUCTURE.md (Directory Tree)

**How the RAG pipeline works:**
- Quick overview: QUICK_REFERENCE.md (Core Workflows)
- Deep dive: CODEBASE_OVERVIEW.md (Section 2.1: RAG Pipeline)

**How evaluation works:**
- Quick overview: QUICK_REFERENCE.md (Evaluation Methods)
- Deep dive: CODEBASE_OVERVIEW.md (Section 2.3: Evaluation Framework)

**How agents communicate (A2A):**
- Overview: CODEBASE_OVERVIEW.md (Section 3: Agent-to-Agent System)
- File locations: FILE_STRUCTURE.md (A2A Agent Framework)

**How to run the system:**
- Quick start: QUICK_REFERENCE.md (Quick Start section)
- Detailed commands: CODEBASE_OVERVIEW.md (Section 6: CLI Interfaces)

**How to extend the system:**
- Options: CODEBASE_OVERVIEW.md (Section 11: Extension Points)
- Quick examples: QUICK_REFERENCE.md (Extending section)

**Where specific files are located:**
- Use: FILE_STRUCTURE.md (Key Absolute File Paths)
- Or: QUICK_REFERENCE.md (Project Structure section)

**Specific API methods:**
- Use: QUICK_REFERENCE.md (Key Methods & APIs)
- Full docs: CODEBASE_OVERVIEW.md (Section 2: Key Components)

**Troubleshooting:**
- Use: QUICK_REFERENCE.md (Troubleshooting table)
- Or: CODEBASE_OVERVIEW.md (Section 10: Current Limitations)

## Quick Links

### Core Components

| Component | CODEBASE_OVERVIEW | QUICK_REF | FILE_STRUCTURE |
|-----------|-------------------|-----------|-----------------|
| RAG Pipeline | 2.1 | API section | RAG Pipeline section |
| Vector DB | 2.2 | API section | Vector Database section |
| Evaluation | 2.3 | Evaluation section | Evaluation Framework section |
| Green Agent | 2.5 | API section | Green Agent section |
| LLM Client | 2.4 | API section | LLM Client section |
| A2A Framework | 3 | Core Workflows | A2A Framework section |

### Setup & Usage

| Task | CODEBASE_OVERVIEW | QUICK_REF | FILE_STRUCTURE |
|------|-------------------|-----------|-----------------|
| Quick start | 13 | Quick Start | Import Structure |
| CLI usage | 6 | Common Commands | File Statistics |
| Configuration | 8 | Configuration | Configuration Files |
| Adding data | 11 | Extending | Data Files |
| Adding LLM provider | 11 | Extending | LLM Client section |

## File Statistics Summary

```
Total Documentation:       1,412 lines
Total Python Code:        ~3,025 lines
Total Project Size:        ~120 KB (code)

Documentation Breakdown:
- CODEBASE_OVERVIEW.md:  763 lines (54%)
- FILE_STRUCTURE.md:     371 lines (26%)
- QUICK_REFERENCE.md:    278 lines (20%)
```

## How to Use These Documents

### For Development
1. Read QUICK_REFERENCE.md first (10 min)
2. Understand the overview in CODEBASE_OVERVIEW.md (20 min)
3. Refer to FILE_STRUCTURE.md when you need specific files

### For Debugging
1. Check QUICK_REFERENCE.md Troubleshooting table
2. Look up file locations in FILE_STRUCTURE.md
3. Find detailed explanations in CODEBASE_OVERVIEW.md

### For Extension/Customization
1. Review extension points in CODEBASE_OVERVIEW.md (Section 11)
2. Find file locations in FILE_STRUCTURE.md
3. Look up API methods in QUICK_REFERENCE.md

### For Learning
1. Start with QUICK_REFERENCE.md (Project Structure)
2. Read CODEBASE_OVERVIEW.md Section 1 (Architecture)
3. Deep dive into specific sections as needed

## Document Features

### CODEBASE_OVERVIEW.md
- Diagrams and workflow illustrations
- Detailed code examples
- Architectural diagrams (Section 14)
- Comprehensive feature list
- Performance characteristics
- Execution flow diagrams

### QUICK_REFERENCE.md
- Quick lookup tables
- Common command examples
- API reference tables
- Performance benchmarks
- Troubleshooting checklist
- Extension examples

### FILE_STRUCTURE.md
- Complete file tree
- File-by-file breakdown
- Key absolute paths
- Class and method listings
- Import structure
- Git repository info

## Key Concepts Explained

### RAG Pipeline (4 Steps)
1. RETRIEVAL: TF-IDF search finds relevant chunks
2. AUGMENTATION: Context preparation with metadata
3. GENERATION: LLM creates response
4. RESPONSE: Return answer to user

### Evaluation Methods
- **Rule-Based**: Fast substring matching (no API)
- **LLM-Judge**: Semantic evaluation by trusted LLM

### A2A Protocol
- Agent-to-Agent communication standard
- Green Agent (evaluator) ↔ White Agent (executor)
- HTTP-based message protocol

## External Resources

- **GitHub Repository**: https://github.com/NeeralBhalgat/AIPolicyBench
- **Main README**: `/home/momoway/AIPolicyBench/README.md` (Original project docs)
- **Agentify Example**: `/home/momoway/AIPolicyBench/agentify-example-tau-bench/README.md`

## Version Information

- **Created**: November 12, 2025
- **Documentation Version**: 1.0
- **Codebase State**: Latest (main branch)
- **Last Updated**: 2025-11-12

## Summary of Core Files

### Absolute Paths to Key Components

**RAG System:**
- `/home/momoway/AIPolicyBench/safety_datasets_rag.py` - RAG pipeline
- `/home/momoway/AIPolicyBench/simple_vector_db.py` - Vector database

**Agents:**
- `/home/momoway/AIPolicyBench/green_agent/agent.py` - Query interface
- `/home/momoway/AIPolicyBench/green_agent/evaluation.py` - Evaluation

**Utilities:**
- `/home/momoway/AIPolicyBench/utils/llm_client.py` - LLM integration
- `/home/momoway/AIPolicyBench/scripts/parse_pdfs_to_json.py` - PDF parser

**Data:**
- `/home/momoway/AIPolicyBench/data/safety_datasets.json` - Document chunks
- `/home/momoway/AIPolicyBench/data/predefined_queries.json` - Test queries

**A2A Framework:**
- `/home/momoway/AIPolicyBench/agentify-example-tau-bench/main.py` - Entry point
- `/home/momoway/AIPolicyBench/agentify-example-tau-bench/src/launcher.py` - Coordinator

## Getting Started

1. **Read first**: QUICK_REFERENCE.md (10 minutes)
2. **Then read**: CODEBASE_OVERVIEW.md Section 1 (10 minutes)
3. **Then**: Run the setup in QUICK_REFERENCE.md (Quick Start)
4. **For details**: Refer to specific sections as needed

## Questions?

Each document is comprehensive but organized for different use cases:
- **"How do I..."**: Check QUICK_REFERENCE.md
- **"How does..."**: Check CODEBASE_OVERVIEW.md
- **"Where is..."**: Check FILE_STRUCTURE.md

---

Last Updated: November 12, 2025
Documentation Quality: Comprehensive (3 interconnected guides)
Code Coverage: 100% of main components documented
=======
### Documentation Index (Concise)

Start Here
- QUICK_COMMANDS.md: copy/paste commands to run agents, evaluate, reproduce

Core Docs
- ARCHITECTURE.md: high‑level architecture, data flow, key files
- LLM_JUDGE.md: using LLM‑as‑a‑judge (short)
- BENCHMARK_NOTES.md: implementation notes, updates, model config

What You’ll Do Most
- Build vector DB (once):
  - python simple_vector_db.py --json_file data/safety_datasets.json --save
- Run white agent (RAG):
  - python main.py white --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --model deepseek-chat
- Run green agent (evaluator):
  - python main.py green
- One‑shot evaluation:
  - python main.py launch --vector-db ./vector_db/safety_datasets_tfidf_db.pkl --white-model deepseek-chat

LLM‑Judge Option
- Add --llm-judge to launch; ensure API keys in .env

Key Files
- RAG: safety_datasets_rag.py
- Vector DB: simple_vector_db.py
- Evaluators: green_agent/evaluation.py
- A2A green: green_agent/a2a_evaluator.py
- A2A white: white_agent/agent.py
- A2A utils: utils/a2a_client.py

Data & Results
- Queries: data/predefined_queries.json
- Vector DB: vector_db/safety_datasets_tfidf_db.pkl
- Results: results/<model>/(summary.json, statistics.txt)

Troubleshooting
- Bad flags: use hyphenated Typer options (e.g., --vector-db)
- DB missing: rebuild via simple_vector_db.py --save
- Agents not ready: check ports 9001/9002
>>>>>>> 66d7a65b6abc8b3d7ea63a4821ddc611bc7508f2
