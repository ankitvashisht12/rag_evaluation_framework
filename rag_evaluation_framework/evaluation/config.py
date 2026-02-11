from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict

from rag_evaluation_framework.evaluation.chunker.base import Chunker
from rag_evaluation_framework.evaluation.embedder.base import Embedder
from rag_evaluation_framework.evaluation.metrics.base import Metrics
from rag_evaluation_framework.evaluation.reranker.base import Reranker


class EvaluationConfig(BaseModel):
    experiment_prefix: str = ""
    description: str = ""
    max_concurrency: int = 4
    save_results: bool = False
    save_results_path: str = ""


class SweepConfig(BaseModel):
    """Configuration for running a hyperparameter sweep over multiple RAG configurations.

    All list parameters are optional. If not provided, the framework defaults are used.
    The sweep generates a Cartesian product of all provided parameters and runs each
    combination as a separate experiment. Combinations sharing the same (chunker, embedder)
    pair reuse the chunked and embedded knowledge base to avoid redundant API calls.

    Args:
        chunkers: List of chunker instances to evaluate.
        embedders: List of embedder instances to evaluate.
        k_values: List of k (top-k retrieval) values to evaluate.
        rerankers: List of reranker instances (use None for no reranking).
        metrics: Metrics to compute for every experiment. If not provided, all default
                 token-level metrics are used.
        max_concurrency: Max concurrent queries within each LangSmith evaluation run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunkers: Optional[List[Chunker]] = None
    embedders: Optional[List[Embedder]] = None
    k_values: Optional[List[int]] = None
    rerankers: Optional[List[Optional[Reranker]]] = None
    metrics: Optional[Dict[str, Metrics]] = None
    max_concurrency: int = 4
