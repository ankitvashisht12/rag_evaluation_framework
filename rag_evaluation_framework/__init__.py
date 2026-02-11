import logging

from .evaluation.base_eval import Evaluation
from .evaluation.config import EvaluationConfig, SweepConfig
from .visualization.comparison import ComparisonGraph

__all__ = ["Evaluation", "EvaluationConfig", "SweepConfig", "ComparisonGraph"]

# Set up logging for the library
# NullHandler prevents "No handler found" warnings and silences logs by default
# Users can enable logging by configuring the logger:
#   logging.getLogger("rag_evaluation_framework").setLevel(logging.DEBUG)
logging.getLogger(__name__).addHandler(logging.NullHandler())