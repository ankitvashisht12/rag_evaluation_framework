# RAG Evaluation Framework

A framework for systematically evaluating RAG (Retrieval-Augmented Generation) pipelines using Langsmith SDK.

## Pipeline

1. Pre-processing Data (kb aka knowledge base)
2. Synthetic Data Generation
3. Chunking Strategy
4. Embedding model
    - 4.1 Custom Embedding model (for adding vector store or db)
5. @k parameter aka retrieved documents
6. Re-ranker (optional)

## Single Evaluation

```python
from rag_evaluation_framework import Evaluation

evaluation = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

results = evaluation.run(
    chunker=my_chunker,
    embedder=my_embedder,
    vector_store=my_vector_store,  # optional, defaults to Chroma
    k=5,
    reranker=my_reranker  # optional
)
```

## Hyperparameter Sweep

Run multiple configurations at once and compare results:

```python
from rag_evaluation_framework import Evaluation, SweepConfig
from rag_evaluation_framework.evaluation.chunker import RecursiveCharTextSplitter
from rag_evaluation_framework.evaluation.embedder.openai_embedder import OpenAIEmbedder

evaluation = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

sweep_results = evaluation.sweep(
    sweep_config=SweepConfig(
        chunkers=[
            RecursiveCharTextSplitter(chunk_size=500, chunk_overlap=50),
            RecursiveCharTextSplitter(chunk_size=1000, chunk_overlap=100),
        ],
        embedders=[
            OpenAIEmbedder(model_name="text-embedding-3-small"),
            OpenAIEmbedder(model_name="text-embedding-3-large"),
        ],
        k_values=[5, 10, 20],
        rerankers=[None],
    )
)
```

Combinations sharing the same `(chunker, embedder)` pair reuse the chunked and embedded knowledge base, so you don't pay for redundant embedding API calls.

## Visualization

```python
from rag_evaluation_framework import ComparisonGraph

graph = ComparisonGraph(sweep_results)
graph.bar()           # grouped bar chart
graph.line(x="k")     # line chart varying k
graph.heatmap()       # colour-coded grid
```

## Documentation

See the [docs/](docs/) folder for detailed guides:

- [Main overview](docs/main.md)
- [Evaluation & Sweep](docs/evaluation.md)
- [Metrics](docs/metrics.md)
- [Visualization](docs/visualization.md)
