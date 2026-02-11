import frontmatter
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from langsmith import evaluate

from rag_evaluation_framework.evaluation.chunker.base import Chunker
from rag_evaluation_framework.evaluation.metrics.base import Metrics
from rag_evaluation_framework.evaluation.vector_store.base import VectorStore
from rag_evaluation_framework.evaluation.reranker.base import Reranker
from rag_evaluation_framework.evaluation.embedder.base import Embedder
from rag_evaluation_framework.evaluation.config import EvaluationConfig, SweepConfig
from rag_evaluation_framework.evaluation.utils import get_langsmith_evaluators
from rag_evaluation_framework.evaluation.embedder.openai_embedder import OpenAIEmbedder
from rag_evaluation_framework.evaluation.chunker.recursive_char_text_splitter import RecursiveCharTextSplitter
from rag_evaluation_framework.evaluation.vector_store.chroma import ChromaVectorStore
from rag_evaluation_framework.evaluation.metrics.token_level_recall import TokenLevelRecall
from rag_evaluation_framework.evaluation.metrics.token_level_precision import TokenLevelPrecision
from rag_evaluation_framework.evaluation.metrics.token_level_iou import TokenLevelIoU
from rag_evaluation_framework.evaluation.metrics.token_level_precision_omega import TokenLevelPrecisionOmega
from rag_evaluation_framework.evaluation.metrics.token_level_mrr import TokenLevelMRR

# Get logger for this module
logger = logging.getLogger(__name__)


class Evaluation:
    """
    RAG Evaluation Framework for systematically evaluating retrieval pipelines.

    This class orchestrates the evaluation process:
    1. Load and chunk knowledge base documents
    2. Embed chunks and store in vector store
    3. Run retrieval against a LangSmith dataset
    4. Calculate metrics and return results

    Example:
        >>> from rag_evaluation_framework import Evaluation
        >>> evaluator = Evaluation(
        ...     langsmith_dataset_name="my-dataset",
        ...     kb_data_path="./knowledge_base"
        ... )
        >>> results = evaluator.run(k=5)
    """

    langsmith_dataset_name: str
    kb_data_path: str
    query_field: str

    def __init__(
        self,
        langsmith_dataset_name: str,
        kb_data_path: str,
        query_field: str = "question",
    ):
        """
        Initialize the Evaluation framework.

        Args:
            langsmith_dataset_name: Name of the LangSmith dataset to evaluate against.
            kb_data_path: Path to directory containing knowledge base markdown files.
            query_field: Field name in the dataset containing the query/question.
                        Defaults to "question". Common alternatives: "query", "input".
        """
        self.langsmith_dataset_name = langsmith_dataset_name
        self.kb_data_path = kb_data_path
        self.query_field = query_field

        logger.debug(
            "Initialized Evaluation with dataset='%s', kb_path='%s', query_field='%s'",
            langsmith_dataset_name,
            kb_data_path,
            query_field,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def __get_kb_files_path(self) -> List[Path]:
        if not os.path.exists(self.kb_data_path):
            logger.error("Knowledge base path does not exist: %s", self.kb_data_path)
            raise FileNotFoundError(
                f"Knowledge base data path {self.kb_data_path} does not exist"
            )

        files = [
            Path(os.path.join(self.kb_data_path, file))
            for file in os.listdir(self.kb_data_path)
            if file.endswith(".md")
        ]
        logger.debug("Found %d markdown files in knowledge base", len(files))
        return files

    def __run_retrieval(
        self,
        input: dict,
        embedder: Embedder,
        vector_store: VectorStore,
        k: int,
        reranker: Optional[Reranker] = None,
    ) -> List[dict]:
        """Run retrieval for a single query."""
        query = input.get(self.query_field, "")

        if not query:
            logger.warning("Empty query received, returning empty results")
            return []

        logger.debug(
            "Running retrieval for query: %s...",
            query[:50] if len(query) > 50 else query,
        )

        query_embedding = embedder.embed_docs([query])[0]
        retrieved_chunks = vector_store.search(query_embedding, k)
        logger.debug("Retrieved %d chunks from vector store", len(retrieved_chunks))

        if reranker:
            logger.debug("Applying reranker")
            chunk_texts = [chunk.get("text", "") for chunk in retrieved_chunks]
            reranked_texts = reranker.rerank(chunk_texts, query, k)
            chunks_by_text: Dict[str, List[dict]] = {}
            for chunk in retrieved_chunks:
                chunks_by_text.setdefault(chunk.get("text", ""), []).append(chunk)
            reranked_chunks: List[dict] = []
            for text in reranked_texts:
                if text in chunks_by_text and chunks_by_text[text]:
                    reranked_chunks.append(chunks_by_text[text].pop(0))
            retrieved_chunks = reranked_chunks

        return retrieved_chunks

    # ------------------------------------------------------------------ #
    #  Default component factories                                         #
    # ------------------------------------------------------------------ #

    def __get_default_chunker(self) -> Chunker:
        return RecursiveCharTextSplitter(chunk_size=100, chunk_overlap=10)

    def __get_default_embedder(self) -> Embedder:
        return OpenAIEmbedder(model_name="text-embedding-3-small")

    def __get_default_vector_store(self) -> VectorStore:
        return ChromaVectorStore()

    def __get_default_metrics(self) -> Dict[str, Metrics]:
        return {
            "token_level_recall": TokenLevelRecall(),
            "token_level_precision": TokenLevelPrecision(),
            "token_level_iou": TokenLevelIoU(),
            "token_level_precision_omega": TokenLevelPrecisionOmega(),
            "token_level_mrr": TokenLevelMRR(),
        }

    # ------------------------------------------------------------------ #
    #  Internal pipeline stages (used by both run and sweep)               #
    # ------------------------------------------------------------------ #

    def _process_kb(
        self, chunker: Chunker, embedder: Embedder
    ) -> List[Dict[str, Any]]:
        """Chunk and embed all KB documents.

        Returns a list of batches (one per file), each containing texts,
        embeddings, metadatas, and doc_ids ready to be indexed.
        """
        kb_markdown_files_path = self.__get_kb_files_path()
        processed: List[Dict[str, Any]] = []
        total_chunks = 0

        for file_path in kb_markdown_files_path:
            logger.debug("Processing file: %s", file_path.name)
            with open(file_path, "r", encoding="utf-8") as file:
                post = frontmatter.load(file)
                markdown_content = post.content
                metadata_value = dict(post.metadata) if post.metadata else {}

                doc_id = file_path.name
                base_metadata = {
                    k: v
                    for k, v in metadata_value.items()
                    if isinstance(v, (str, int, float, bool))
                }
                base_metadata["doc_id"] = doc_id

                chunked_docs = chunker.chunk(markdown_content)
                logger.debug(
                    "Created %d chunks from %s", len(chunked_docs), file_path.name
                )
                chunk_texts = [chunk.text for chunk in chunked_docs]
                embeddings = embedder.embed_docs(chunk_texts)
                metadatas = [
                    {
                        **base_metadata,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                    }
                    for chunk in chunked_docs
                ]
                doc_ids = [str(uuid.uuid4()) for _ in chunked_docs]

                processed.append(
                    {
                        "texts": chunk_texts,
                        "embeddings": embeddings,
                        "metadatas": metadatas,
                        "doc_ids": doc_ids,
                    }
                )
                total_chunks += len(chunked_docs)

        logger.info(
            "Knowledge base processed: %d total chunks from %d files",
            total_chunks,
            len(kb_markdown_files_path),
        )
        return processed

    def _index_kb(
        self, processed_kb: List[Dict[str, Any]], vector_store: VectorStore
    ) -> None:
        """Load pre-processed KB data into a vector store."""
        for batch in processed_kb:
            vector_store.add_docs(
                batch["texts"],
                batch["embeddings"],
                doc_ids=batch["doc_ids"],
                metadatas=batch["metadatas"],
            )

    def _evaluate_retrieval(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        metrics: Dict[str, Metrics],
        k: int,
        reranker: Optional[Reranker],
        config: Optional[EvaluationConfig],
    ) -> Dict[str, Any]:
        """Run retrieval evaluation against the LangSmith dataset and extract metrics."""
        langsmith_evaluators = get_langsmith_evaluators(metrics, k)

        experiment_prefix = config.experiment_prefix if config else ""
        description = config.description if config else ""
        max_concurrency = config.max_concurrency if config else 4

        logger.info(
            "Running LangSmith evaluation on dataset '%s' with %d evaluators",
            self.langsmith_dataset_name,
            len(langsmith_evaluators),
        )

        results = evaluate(
            lambda input: self.__run_retrieval(
                input, embedder, vector_store, k, reranker
            ),
            data=self.langsmith_dataset_name,
            evaluators=langsmith_evaluators,
            experiment_prefix=experiment_prefix,
            description=description,
            max_concurrency=max_concurrency,
        )

        logger.debug("LangSmith evaluation completed")

        # ----- extract metrics and experiment URL from results ----- #
        metrics_dict: Dict[str, float] = {}
        langsmith_experiment_url = None

        if hasattr(results, "experiment_url"):
            langsmith_experiment_url = results.experiment_url
        elif hasattr(results, "experiment_name"):
            pass

        try:
            metrics_by_name: Dict[str, List[float]] = {}

            for row in results:
                if isinstance(row, dict):
                    evaluation_results = row.get("evaluation_results")
                    if isinstance(evaluation_results, dict):
                        results_list = evaluation_results.get("results")
                        if isinstance(results_list, list):
                            for result in results_list:
                                if hasattr(result, "key") and hasattr(
                                    result, "score"
                                ):
                                    metric_name = (
                                        str(result.key) if result.key else None
                                    )
                                    score = result.score
                                elif isinstance(result, dict):
                                    metric_name = (
                                        str(result.get("key"))
                                        if result.get("key")
                                        else None
                                    )
                                    score = result.get("score")
                                else:
                                    metric_name = None
                                    score = None

                                if metric_name and isinstance(score, (int, float)):
                                    metrics_by_name.setdefault(
                                        metric_name, []
                                    ).append(float(score))

                    run = row.get("run")
                else:
                    run = row

                if hasattr(run, "feedback") and run.feedback:
                    feedback_list = run.feedback
                    if isinstance(feedback_list, (list, tuple)):
                        for feedback in feedback_list:
                            if hasattr(feedback, "key") and hasattr(
                                feedback, "score"
                            ):
                                metric_name = (
                                    str(feedback.key) if feedback.key else None
                                )
                                score = feedback.score
                                if metric_name and isinstance(score, (int, float)):
                                    if metric_name not in metrics_by_name:
                                        metrics_by_name[metric_name] = []
                                    metrics_by_name[metric_name].append(float(score))

                if hasattr(run, "feedback_stats") and run.feedback_stats:
                    feedback_stats = run.feedback_stats
                    if isinstance(feedback_stats, dict):
                        for metric_name_key, score_value in feedback_stats.items():
                            metric_name = (
                                str(metric_name_key) if metric_name_key else None
                            )
                            if not metric_name:
                                continue
                            if metric_name not in metrics_by_name:
                                metrics_by_name[metric_name] = []
                            if isinstance(score_value, (list, tuple)):
                                float_values = [
                                    float(v)
                                    for v in score_value
                                    if isinstance(v, (int, float))
                                ]
                                metrics_by_name[metric_name].extend(float_values)
                            elif isinstance(score_value, (int, float)):
                                metrics_by_name[metric_name].append(float(score_value))

            for metric_name, scores in metrics_by_name.items():
                if scores:
                    metrics_dict[metric_name] = sum(scores) / len(scores)

        except (AttributeError, TypeError, StopIteration) as e:
            logger.warning("Could not extract metrics from results: %s", str(e))
            try:
                if hasattr(results, "to_pandas"):
                    df = results.to_pandas()
                    logger.debug("Falling back to pandas extraction")
            except Exception as fallback_error:
                logger.debug(
                    "Pandas fallback also failed: %s", str(fallback_error)
                )

        logger.info("Evaluation complete. Metrics: %s", metrics_dict)

        return {
            "metrics": metrics_dict,
            "langsmith_experiment_url": langsmith_experiment_url,
            "raw_results": results,
        }

    # ------------------------------------------------------------------ #
    #  Sweep helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_component_label(component: Any, component_type: str) -> str:
        """Generate a human-readable label for a pipeline component."""
        if component is None:
            return "None"
        name = type(component).__name__
        if component_type == "chunker":
            if hasattr(component, "chunk_size") and hasattr(
                component, "chunk_overlap"
            ):
                return f"{name}({component.chunk_size},{component.chunk_overlap})"
        elif component_type == "embedder":
            if hasattr(component, "model_name"):
                return f"{name}({component.model_name})"
        elif component_type == "reranker":
            if hasattr(component, "model_name"):
                return f"{name}({component.model_name})"
        return name

    def _generate_sweep_prefix(
        self,
        chunker: Chunker,
        embedder: Embedder,
        k: int,
        reranker: Optional[Reranker],
    ) -> str:
        """Generate a unique experiment prefix for one sweep combination."""
        parts = [
            self._get_component_label(chunker, "chunker"),
            self._get_component_label(embedder, "embedder"),
            f"k={k}",
        ]
        if reranker is not None:
            parts.append(self._get_component_label(reranker, "reranker"))
        return "-".join(parts)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        metrics: Optional[Dict[str, Metrics]] = None,
        k: int = 5,
        reranker: Optional[Reranker] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> Dict[str, Any]:
        """Run a single evaluation with the given configuration.

        Args:
            chunker: Document chunker (default: RecursiveCharTextSplitter 100/10).
            embedder: Embedding model (default: OpenAI text-embedding-3-small).
            vector_store: Vector database (default: in-memory ChromaDB).
            metrics: Dict of metric name → Metrics instance.
            k: Number of documents to retrieve per query.
            reranker: Optional reranker to apply after retrieval.
            config: Experiment configuration (prefix, description, concurrency).

        Returns:
            Dict with keys ``metrics``, ``langsmith_experiment_url``, ``raw_results``.
        """
        if not self.langsmith_dataset_name:
            raise ValueError("langsmith_dataset_name is required")

        if not self.kb_data_path:
            raise ValueError("kb_data_path is required")

        logger.info("Starting evaluation run with k=%d", k)

        if not chunker:
            chunker = self.__get_default_chunker()
            logger.debug("Using default chunker: %s", type(chunker).__name__)

        if not embedder:
            embedder = self.__get_default_embedder()
            logger.debug("Using default embedder: %s", type(embedder).__name__)

        if not vector_store:
            vector_store = self.__get_default_vector_store()
            logger.debug("Using default vector store: %s", type(vector_store).__name__)

        if not metrics:
            metrics = self.__get_default_metrics()
            logger.debug("Using default metrics: %s", list(metrics.keys()))

        processed_kb = self._process_kb(chunker, embedder)
        self._index_kb(processed_kb, vector_store)
        return self._evaluate_retrieval(
            embedder, vector_store, metrics, k, reranker, config
        )

    def sweep(self, sweep_config: SweepConfig) -> List[Dict[str, Any]]:
        """Run a hyperparameter sweep over multiple RAG configurations.

        Generates the Cartesian product of all provided parameters and runs each
        combination as a separate LangSmith experiment.  Combinations that share
        the same (chunker, embedder) pair reuse chunked and embedded data so that
        expensive embedding API calls are not repeated.

        Args:
            sweep_config: A ``SweepConfig`` specifying the parameter grid.

        Returns:
            List of result dicts, one per combination.  Each dict contains:

            - ``config`` – dict describing the parameters used for this run
              (``chunker``, ``embedder``, ``k``, ``reranker``).
            - ``metrics`` – dict of metric name → averaged score.
            - ``langsmith_experiment_url`` – URL to the LangSmith experiment.
            - ``raw_results`` – the raw LangSmith ``ExperimentResults`` object.
        """
        if not self.langsmith_dataset_name:
            raise ValueError("langsmith_dataset_name is required")
        if not self.kb_data_path:
            raise ValueError("kb_data_path is required")

        chunkers = sweep_config.chunkers or [self.__get_default_chunker()]
        embedders = sweep_config.embedders or [self.__get_default_embedder()]
        k_values = sweep_config.k_values or [5]
        rerankers = sweep_config.rerankers or [None]
        metrics = sweep_config.metrics or self.__get_default_metrics()

        total_combos = len(chunkers) * len(embedders) * len(k_values) * len(rerankers)
        logger.info("Starting sweep with %d total combinations", total_combos)

        all_results: List[Dict[str, Any]] = []
        combo_idx = 0

        for chunker in chunkers:
            for embedder in embedders:
                chunker_label = self._get_component_label(chunker, "chunker")
                embedder_label = self._get_component_label(embedder, "embedder")

                logger.info(
                    "Processing KB for chunker=%s, embedder=%s",
                    chunker_label,
                    embedder_label,
                )
                processed_kb = self._process_kb(chunker, embedder)

                for k in k_values:
                    for reranker in rerankers:
                        combo_idx += 1
                        reranker_label = (
                            self._get_component_label(reranker, "reranker")
                            if reranker is not None
                            else None
                        )
                        prefix = self._generate_sweep_prefix(
                            chunker, embedder, k, reranker
                        )

                        logger.info(
                            "Sweep [%d/%d]: %s", combo_idx, total_combos, prefix
                        )

                        # Fresh vector store with a unique collection name to
                        # avoid dimension conflicts across different embedders.
                        vector_store = ChromaVectorStore(
                            collection_name=f"sweep_{combo_idx}"
                        )
                        self._index_kb(processed_kb, vector_store)

                        config = EvaluationConfig(
                            experiment_prefix=prefix,
                            description=f"Sweep: {prefix}",
                            max_concurrency=sweep_config.max_concurrency,
                        )

                        result = self._evaluate_retrieval(
                            embedder, vector_store, metrics, k, reranker, config
                        )

                        result["config"] = {
                            "chunker": chunker_label,
                            "embedder": embedder_label,
                            "k": k,
                            "reranker": reranker_label,
                        }

                        all_results.append(result)

        logger.info("Sweep complete: %d experiments ran", len(all_results))
        return all_results
