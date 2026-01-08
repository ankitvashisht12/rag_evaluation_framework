# Evaluation

Core evaluation pipeline for RAG systems using Langsmith SDK.

## Single Evaluation

Run a single evaluation with specific configuration:

```python
from rag_evaluation_framework import Evaluation

evaluator = Evaluation()

results = evaluator.run(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base",
    chunker=my_chunker,           # See [Chunker](chunker.md)
    embedding_func=my_embedder,   # See [Embedder](embedder.md)
    k=5,
    reranker=my_reranker,         # Optional
)
```

### Process

1. Load knowledge base documents from `kb_data_path`
2. Chunk documents using provided chunker
3. Embed chunks using embedder and store in vector database
4. Fetch evaluation dataset from Langsmith
5. Run retrieval for each query in dataset
6. Calculate metrics (see [Metrics](metrics.md)): recall@k, precision@k, MRR@k
7. Return results with Langsmith trace URLs

## Hyperparameter Sweep

Evaluate multiple configurations automatically:

```python
from rag_evaluation_framework import Evaluation, SweepConfig

evaluator = Evaluation()

sweep_results = evaluator.sweep(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base",
    sweep_config=SweepConfig(
        chunkers=[chunker1, chunker2],
        embedders=[embedder1, embedder2],
        k_values=[5, 10, 20],
        rerankers=[None, reranker1],
    )
)
```

### How It Works

- Generates all combinations of provided parameters
- Runs each combination as a separate Langsmith experiment
- Collects all results with metadata about each configuration
- Returns `SweepResults` object for comparison and visualization

## Components

- **[Chunker](chunker.md)** - Document chunking strategies
- **[Embedder](embedder.md)** - Embedding model integration
- **[Metrics](metrics.md)** - Evaluation metrics (recall, precision, MRR)
- **[Vector Store](vector_store.md)** - Vector database abstraction

## Results

Evaluation results include:
- Metrics per k value (recall@k, precision@k, MRR@k)
- Langsmith experiment URLs
- Retrieved documents for each query
- Configuration metadata

Results can be saved and loaded for later analysis and visualization.

