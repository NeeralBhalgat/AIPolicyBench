"""Main entry point for running green_agent as a module."""

from .a2a_evaluator import start_green_agent

if __name__ == "__main__":
    # Start green agent with default settings
    # AgentBeats will provide AGENT_URL via environment variable
    start_green_agent(
        agent_name="green_agent",
        host="0.0.0.0",  # Listen on all interfaces for container deployment
        port=9001
    )
