import frontmatter
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from langsmith import evaluate

from rag_evaluation_framework.evaluation.chunker.base import Chunker
from rag_evaluation_framework.evaluation.metrics.base import Metrics
from rag_evaluation_framework.evaluation.vector_store.base import VectorStore
from rag_evaluation_framework.evaluation.reranker.base import Reranker
from rag_evaluation_framework.evaluation.embedder.base import Embedder
from rag_evaluation_framework.evaluation.config import EvaluationConfig
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
        query_field: str = "question"
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
            langsmith_dataset_name, kb_data_path, query_field
        )

    def __get_kb_files_path(self) -> List[Path]:
        if not os.path.exists(self.kb_data_path):
            logger.error("Knowledge base path does not exist: %s", self.kb_data_path)
            raise FileNotFoundError(f"Knowledge base data path {self.kb_data_path} does not exist")

        files = [
            Path(os.path.join(self.kb_data_path, file))
            for file in os.listdir(self.kb_data_path)
            if file.endswith(".md")
        ]
        logger.debug("Found %d markdown files in knowledge base", len(files))
        return files

    def __run_retrieval(self, input: dict, embedder: Embedder, vector_store: VectorStore, k: int, reranker: Optional[Reranker] = None) -> List[dict]:
        """
        Run retrieval for a single query.
        
        Args:
            input: Dictionary containing the query (field name specified by query_field)
            embedder: Embedder to use for query embedding
            vector_store: Vector store to search
            k: Number of results to retrieve
            reranker: Optional reranker to apply
            
        Returns:
            List of retrieved chunks with text and metadata
        """
        query = input.get(self.query_field, "")
        
        if not query:
            logger.warning("Empty query received, returning empty results")
            return []
        
        logger.debug("Running retrieval for query: %s...", query[:50] if len(query) > 50 else query)
        
        # Embed the query
        query_embedding = embedder.embed_docs([query])[0]
        
        # Search in vector store using query embedding
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

    def __get_default_chunker(self) -> Chunker:
        return RecursiveCharTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
        )
        

    def __get_default_embedder(self) -> Embedder:
        return OpenAIEmbedder(
            model_name="text-embedding-3-small",
        )

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

        # Process Knowledge base (chunk, embed and store in vector store)
        logger.info("Processing knowledge base...")
        kb_markdown_files_path = self.__get_kb_files_path()
        total_chunks = 0

        for file_path in kb_markdown_files_path:
            logger.debug("Processing file: %s", file_path.name)
            with open(file_path, "r", encoding="utf-8") as file:
                post = frontmatter.load(file)
                markdown_content = post.content
                metadata_value = dict(post.metadata) if post.metadata else {}

                doc_id = file_path.name
                base_metadata = {
                    k: v for k, v in metadata_value.items()
                    if isinstance(v, (str, int, float, bool))
                }
                base_metadata["doc_id"] = doc_id

                chunked_docs = chunker.chunk(markdown_content)
                logger.debug("Created %d chunks from %s", len(chunked_docs), file_path.name)
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
                vector_store.add_docs(
                    chunk_texts,
                    embeddings,
                    doc_ids=doc_ids,
                    metadatas=metadatas,
                )
                total_chunks += len(chunked_docs)

        logger.info("Knowledge base indexed: %d total chunks from %d files", total_chunks, len(kb_markdown_files_path))
        
        langsmith_evaluators = get_langsmith_evaluators(metrics, k)

        # Use config if provided, otherwise use defaults
        experiment_prefix = config.experiment_prefix if config else ""
        description = config.description if config else ""
        max_concurrency = config.max_concurrency if config else 4

        logger.info(
            "Running LangSmith evaluation on dataset '%s' with %d evaluators",
            self.langsmith_dataset_name, len(langsmith_evaluators)
        )
        
        # Run evaluation on langsmith dataset
        results = evaluate(
            lambda input: self.__run_retrieval(input, embedder, vector_store, k, reranker),
            data=self.langsmith_dataset_name,
            evaluators=langsmith_evaluators,
            experiment_prefix=experiment_prefix,
            description=description,
            max_concurrency=max_concurrency,
        )
        
        logger.debug("LangSmith evaluation completed")

        # Extract metrics and experiment URL from results
        # ExperimentResults is iterable - iterate directly over it
        metrics_dict = {}
        langsmith_experiment_url = None
        
        # Try to get experiment URL/name
        if hasattr(results, 'experiment_url'):
            langsmith_experiment_url = results.experiment_url
        elif hasattr(results, 'experiment_name'):
            # Experiment name is available - URL can be constructed if needed
            # LangSmith typically prints the URL, but we store the name for reference
            pass
        
        # Extract metric scores from results
        # Iterate through runs in the experiment results (ExperimentResults is iterable)
        try:
            # Collect all metric scores across all runs
            metrics_by_name: Dict[str, List[float]] = {}
            
            for row in results:
                # Newer langsmith returns ExperimentResultRow dicts
                if isinstance(row, dict):
                    evaluation_results = row.get("evaluation_results")
                    if isinstance(evaluation_results, dict):
                        results_list = evaluation_results.get("results")
                        if isinstance(results_list, list):
                            for result in results_list:
                                if hasattr(result, "key") and hasattr(result, "score"):
                                    metric_name = str(result.key) if result.key else None
                                    score = result.score
                                elif isinstance(result, dict):
                                    metric_name = str(result.get("key")) if result.get("key") else None
                                    score = result.get("score")
                                else:
                                    metric_name = None
                                    score = None

                                if metric_name and isinstance(score, (int, float)):
                                    metrics_by_name.setdefault(metric_name, []).append(float(score))

                    run = row.get("run")
                else:
                    run = row

                # Check for feedback/evaluation results on each run (older API)
                if hasattr(run, 'feedback') and run.feedback:
                    # Feedback is a list of evaluation results
                    feedback_list = run.feedback
                    if isinstance(feedback_list, (list, tuple)):
                        for feedback in feedback_list:
                            if hasattr(feedback, 'key') and hasattr(feedback, 'score'):
                                metric_name = str(feedback.key) if feedback.key else None
                                score = feedback.score
                                # Type check and convert score to float
                                if metric_name and isinstance(score, (int, float)):
                                    # Collect scores for averaging later
                                    if metric_name not in metrics_by_name:
                                        metrics_by_name[metric_name] = []
                                    metrics_by_name[metric_name].append(float(score))
                
                # Alternative: check feedback_stats if available
                if hasattr(run, 'feedback_stats') and run.feedback_stats:
                    feedback_stats = run.feedback_stats
                    if isinstance(feedback_stats, dict):
                        for metric_name_key, score_value in feedback_stats.items():
                            metric_name = str(metric_name_key) if metric_name_key else None
                            if not metric_name:
                                continue
                            if metric_name not in metrics_by_name:
                                metrics_by_name[metric_name] = []
                            # Handle both single values and lists
                            if isinstance(score_value, (list, tuple)):
                                # Convert all values to float
                                float_values = [float(v) for v in score_value if isinstance(v, (int, float))]
                                metrics_by_name[metric_name].extend(float_values)
                            elif isinstance(score_value, (int, float)):
                                metrics_by_name[metric_name].append(float(score_value))
            
            # Calculate average scores for each metric (or use last value)
            # Averaging makes sense for metrics calculated across multiple runs
            for metric_name, scores in metrics_by_name.items():
                if scores:
                    # Use average score across all runs
                    metrics_dict[metric_name] = sum(scores) / len(scores)
                    
        except (AttributeError, TypeError, StopIteration) as e:
            # If results structure is different, try alternative extraction methods
            # The raw_results will still contain all the information
            logger.warning("Could not extract metrics from results: %s", str(e))
            try:
                # Alternative: Use to_pandas() method if available
                if hasattr(results, 'to_pandas'):
                    df = results.to_pandas()
                    logger.debug("Falling back to pandas extraction")
                    # Extract metrics from dataframe columns if they exist
                    # This is a fallback - the iteration method above should work
                    pass
            except Exception as fallback_error:
                logger.debug("Pandas fallback also failed: %s", str(fallback_error))

        logger.info("Evaluation complete. Metrics: %s", metrics_dict)
        
        return {
            "metrics": metrics_dict,
            "langsmith_experiment_url": langsmith_experiment_url,
            "raw_results": results
        }


