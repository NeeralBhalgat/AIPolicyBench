"""Green agent module - evaluation and assessment of RAG agents."""

# Original interface (backward compatible)
from green_agent.agent import PredefinedQueryInterface

# A2A evaluator
from green_agent.a2a_evaluator import start_green_agent, GreenAgentExecutor

# Evaluation utilities
from green_agent.evaluation import RuleBasedEvaluator

__all__ = [
    'PredefinedQueryInterface',
    'start_green_agent',
    'GreenAgentExecutor',
    'RuleBasedEvaluator'
]
