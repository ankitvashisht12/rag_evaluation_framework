import os
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
from rag_evaluation_framework.evaluation.metrics.chunk_level_recall import ChunkLevelRecall

class Evaluation:

    langsmith_dataset_name: str
    kb_data_path: str

    def __init__(self, langsmith_dataset_name: str, kb_data_path: str):
        self.langsmith_dataset_name = langsmith_dataset_name
        self.kb_data_path = kb_data_path

    def __get_kb_markdown_files_path(self) -> List[Path]:
        if not os.path.exists(self.kb_data_path):
            raise FileNotFoundError(f"Knowledge base data path {self.kb_data_path} does not exist")

        return [Path(os.path.join(self.kb_data_path, file)) for file in os.listdir(self.kb_data_path) if file.endswith(".md")]

    def __run_retrieval(self, input: dict, embedder: Embedder, vector_store: VectorStore, k: int, reranker: Optional[Reranker] = None) -> List[str]:
        question = input.get("question", "")
        
        # Embed the query
        query_embedding = embedder.embed_docs([question])[0]
        
        # Search in vector store using query embedding
        retrieved_chunks = vector_store.search(query_embedding, k)

        if reranker:
            retrieved_chunks = reranker.rerank(retrieved_chunks, question, k)

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
            "chunk_level_recall": ChunkLevelRecall()
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

        if not chunker:
            chunker = self.__get_default_chunker()
        
        if not embedder:
            embedder = self.__get_default_embedder()

        if not vector_store:
            vector_store = self.__get_default_vector_store()

        if not metrics:
            metrics= self.__get_default_metrics()

        # Process Knowledge base (chunk, embed and store in vector store)
        kb_markdown_files_path = self.__get_kb_markdown_files_path()

        for file_path in kb_markdown_files_path:
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
                chunked_docs = chunker.chunk(file_content)
                embeddings = embedder.embed_docs(chunked_docs)
                vector_store.add_docs(chunked_docs, embeddings)

        langsmith_evaluators = get_langsmith_evaluators(metrics, k)

        # Use config if provided, otherwise use defaults
        experiment_prefix = config.experiment_prefix if config else ""
        description = config.description if config else ""
        max_concurrency = config.max_concurrency if config else 4

        # Run evaluation on langsmith dataset
        results = evaluate(
            lambda input: self.__run_retrieval(input, embedder, vector_store, k, reranker),
            data=self.langsmith_dataset_name,
            evaluators=langsmith_evaluators,
            experiment_prefix=experiment_prefix,
            description=description,
            max_concurrency=max_concurrency,
        )

        # Extract metrics and experiment URL from results
        metrics_dict = {}
        langsmith_experiment_url = None
        
        # Process results to extract metrics
        if hasattr(results, 'experiment_url'):
            langsmith_experiment_url = results.experiment_url
        
        # Extract metric scores from results
        # The results object structure may vary, so we handle it safely
        try:
            if hasattr(results, 'results') and results.results:
                # Check if results.results is iterable (list or tuple)
                if isinstance(results.results, (list, tuple)):
                    results_list: List[Any] = list(results.results)
                    for result in results_list:
                        if hasattr(result, 'evaluation_results'):
                            eval_results = result.evaluation_results
                            if isinstance(eval_results, (list, tuple)):
                                for eval_result in eval_results:
                                    if hasattr(eval_result, 'key') and hasattr(eval_result, 'score'):
                                        metric_name = eval_result.key
                                        score = eval_result.score
                                        metrics_dict[metric_name] = score
        except (AttributeError, TypeError):
            # If results structure is different, we'll just return empty metrics dict
            # The raw_results will still contain all the information
            pass

        return {
            "metrics": metrics_dict,
            "langsmith_experiment_url": langsmith_experiment_url,
            "raw_results": results
        }


