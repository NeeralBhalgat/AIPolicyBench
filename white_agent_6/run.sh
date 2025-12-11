#!/bin/bash
# AgentBeats Integration - Universal Agent Launcher
# Set AGENT_TYPE environment variable to 'green' or 'white'
# Default to green if not specified

AGENT_TYPE=${AGENT_TYPE:-green}
PORT=${AGENT_PORT:-9001}
MODEL=${WHITE_AGENT_MODEL:-deepseek/deepseek-chat}
USE_DIRECT_LLM=${USE_DIRECT_LLM:-false}

echo "🚀 Starting $AGENT_TYPE agent on port $PORT..."

if [ "$AGENT_TYPE" = "green" ]; then
    # Launch green agent (evaluator)
    python main.py green --host 0.0.0.0 --port $PORT
elif [ "$AGENT_TYPE" = "white" ]; then
    # Launch white agent (RAG or Direct LLM system)
    if [ "$USE_DIRECT_LLM" = "true" ]; then
        echo "   Mode: Direct LLM (no RAG)"
        python main.py white --host 0.0.0.0 --port $PORT --model $MODEL --direct-llm
    else
        echo "   Mode: Web Search RAG"
        python main.py white --host 0.0.0.0 --port $PORT --model $MODEL
    fi
else
    echo "❌ Error: Unknown AGENT_TYPE='$AGENT_TYPE'. Use 'green' or 'white'."
    exit 1
fi
