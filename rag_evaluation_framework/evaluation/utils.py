from typing import List, Dict, Optional, Callable
from langsmith import Evaluator
from rag_evaluation_framework.evaluation.metrics.base import Metrics

def get_langsmith_evaluators(metrics: Dict[str, Metrics], k: Optional[int] = None) -> List[Callable]:
    evaluators = []

    for metric_name, metric_instance in metrics.items():
        evaluators.append(metric_instance.to_langsmith_evaluator(metric_name, k))

    return evaluators

