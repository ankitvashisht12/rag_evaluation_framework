import os
from pathlib import Path
from typing import List, Optional, Dict
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

    def __run_retrieval(self, input: dict, chunker: Chunker, embedder: Embedder, vector_store: VectorStore, k: int, reranker: Optional[Reranker] = None) -> List[str]:

        question = input["question"] # TODO: this can be anything so better take it from the user itself during class instantiation?

        retrieved_chunks = vector_store.search(question, k)

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
        pass

    def __get_default_metrics(self) -> Dict[str, Metrics]:
        pass


    def run(
        self,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        metrics: Optional[Dict[str, Metrics]] = None,
        k: int = 5,
        reranker: Optional[Reranker] = None,
        config: Optional[EvaluationConfig] = None,
    ):
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
            with open(file_path, "r") as file:
                file_content = file.read()
                chunked_docs = chunker.chunk(file_content)
                embeddings = embedder.embed_docs(chunked_docs)
                vector_store.add_docs(chunked_docs, embeddings)

        langsmith_evaluators = get_langsmith_evaluators(metrics)

        # Run evaluation on langsmith dataset
        results = evaluate(
            lambda input: self.__run_retrieval(input, chunker, embedder, vector_store, k, reranker),
            data=self.langsmith_dataset_name,
            evaluators=langsmith_evaluators,
            experiment_prefix="", # TODO: this shoudl come from config
            description="", # TODO: this should from config
            max_concurrency=4, # TODO: this should from config
        )

        # TODO: Ideally it should return { "metrics": { "recall@k": 0.9, "precision@k": 0.9, "mrr@k": 0.9 }, "langsmith_experiment_url": "https://langsmith.com/experiments/1234567890" }
        # and we should have an api to pretty print it in nice format
        return results


