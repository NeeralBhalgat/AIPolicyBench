"""Green agent module - evaluation and assessment of RAG agents."""

# Original interface (backward compatible)
from .agent import PredefinedQueryInterface

# A2A evaluator
from .a2a_evaluator import start_green_agent, GreenAgentExecutor

# Evaluation utilities
from .evaluation import RuleBasedEvaluator

__all__ = [
    'PredefinedQueryInterface',
    'start_green_agent',
    'GreenAgentExecutor',
    'RuleBasedEvaluator'
]
