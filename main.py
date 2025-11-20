#!/usr/bin/env python3
"""CLI entry point for AIPolicyBench with A2A support."""

import typer
import asyncio
from typing import Optional

from green_agent.a2a_evaluator import start_green_agent
from src.white_agent.agent import start_white_agent
from launcher import launch_evaluation

app = typer.Typer(
    help="AIPolicyBench - AI Safety & Policy RAG Agent Benchmark with A2A Support",
    add_completion=False
)


@app.command()
def green(
    host: str = typer.Option("localhost", help="Host to bind to"),
    port: int = typer.Option(9001, help="Port to bind to"),
    agent_name: str = typer.Option("green_agent", help="Agent configuration name")
):
    """Start the green agent (assessment manager/evaluator)."""
    typer.echo("🟢 Starting green agent (evaluator)...")
    start_green_agent(agent_name=agent_name, host=host, port=port)


@app.command()
def white(
    host: str = typer.Option("localhost", help="Host to bind to"),
    port: int = typer.Option(9002, help="Port to bind to"),
    vector_db: str = typer.Option(
        "./vector_db/safety_datasets_tfidf_db.pkl",
        help="Path to vector database (Legacy)"
    ),
    model: str = typer.Option(
        "deepseek-chat",
        help="LLM model to use (e.g., deepseek-chat, mistralai/mistral-7b-instruct, openai/gpt-4o-mini)"
    )
):
    """Start the white agent (Web RAG system being tested)."""
    typer.echo(f"⚪ Starting white agent (Web RAG system) with model: {model}...")
    # We ignore vector_db arg as we are now using Web Search
    start_white_agent(model=model, host=host, port=port)


@app.command()
def launch(
    queries_file: str = typer.Option(
        "data/predefined_queries.json",
        help="Path to predefined queries JSON file"
    ),
    vector_db: str = typer.Option(
        "./vector_db/safety_datasets_tfidf_db.pkl",
        help="Path to vector database"
    ),
    white_model: str = typer.Option(
        "deepseek-chat",
        help="LLM model for white agent (e.g., deepseek-chat, mistralai/mistral-7b-instruct, openai/gpt-4o-mini)"
    ),
    use_llm_judge: bool = typer.Option(
        False,
        "--llm-judge",
        help="Use LLM-as-a-judge evaluation (fixed: gpt-4o-mini)"
    ),
    max_queries: Optional[int] = typer.Option(
        None,
        help="Maximum number of queries to evaluate (default: all queries)"
    ),
    green_host: str = typer.Option("localhost", help="Green agent host"),
    green_port: int = typer.Option(9001, help="Green agent port"),
    white_host: str = typer.Option("localhost", help="White agent host"),
    white_port: int = typer.Option(9002, help="White agent port"),
):
    """Launch the complete A2A evaluation workflow (green + white agents)."""
    typer.echo(f"🚀 Launching complete A2A evaluation...")
    typer.echo(f"   White Agent Model: {white_model}")
    if use_llm_judge:
        typer.echo(f"   LLM Judge Model: gpt-4o-mini (fixed)")
    if max_queries:
        typer.echo(f"   Max Queries: {max_queries}")
    asyncio.run(launch_evaluation(
        queries_file=queries_file,
        vector_db_path=vector_db,
        white_model=white_model,
        use_llm_judge=use_llm_judge,
        max_queries=max_queries,
        green_host=green_host,
        green_port=green_port,
        white_host=white_host,
        white_port=white_port
    ))


@app.command()
def info():
    """Display information about AIPolicyBench."""
    info_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                            AIPolicyBench                                      ║
║                   AI Safety & Policy RAG Agent Benchmark                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 Description:
   AIPolicyBench is a benchmarking system for evaluating RAG (Retrieval-Augmented
   Generation) agents on AI safety and policy questions. It uses the A2A
   (Agent-to-Agent) protocol for standardized agent communication.

🏗️  Architecture:
   • Green Agent: Assessment manager that evaluates RAG agents
   • White Agent: RAG system being tested (answers policy questions)
   • Launcher: Coordinates the evaluation workflow

🎯 Features:
   • A2A protocol support for agent communication
   • Rule-based and LLM-as-a-judge evaluation methods
   • Predefined queries with ground truth answers
   • Comprehensive metrics (correctness, hallucination rate, factuality)
   • Web Search RAG for real-time answers

📝 Usage:
   # Start green agent only:
   python main.py green

   # Start white agent only:
   python main.py white

   # Launch complete evaluation:
   python main.py launch

   # Get help:
   python main.py --help

📚 Documentation:
   • DOCUMENTATION_INDEX.md - Complete documentation index
   • CODEBASE_OVERVIEW.md - Architecture and technical details
   • QUICK_REFERENCE.md - Quick reference guide

🔗 A2A Protocol:
   Agents communicate via HTTP using the A2A (Agent-to-Agent) standard,
   enabling interoperability with other A2A-compliant systems.

    """
    typer.echo(info_text)


@app.command()
def version():
    """Display version information."""
    typer.echo("AIPolicyBench v1.0.0")
    typer.echo("A2A Protocol: Enabled")


if __name__ == "__main__":
    app()
