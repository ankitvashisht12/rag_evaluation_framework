# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) Evaluation Framework designed to systematically evaluate and optimize RAG pipelines through a multi-stage approach.

## Pipeline Architecture

The framework implements a modular pipeline with the following stages:

1. **Pre-processing Data** - Knowledge base (kb) preparation and cleaning
2. **Synthetic Data Generation** - Creating evaluation datasets
3. **Chunking Strategy** - Document segmentation approaches
4. **Embedding Model** - Vector representation of text chunks
   - Supports custom embedding models for different vector stores/databases
5. **Retrieval (@k parameter)** - Number of documents to retrieve
6. **Re-ranker** - Optional re-ranking of retrieved documents

## Development Setup

**Python Version**: 3.12 (specified in .python-version)

**Package Manager**: Using `uv` for dependency management (based on pyproject.toml structure)

**Install Dependencies**:
```bash
uv pip install -e .
```

**Run the Application**:
```bash
python main.py
```

## Project Structure

This is an early-stage project with a minimal structure:
- `main.py` - Entry point (currently a placeholder)
- `pyproject.toml` - Project metadata and dependencies
- `README.md` - Pipeline documentation

## Key Implementation Notes

- The framework is designed to be modular, allowing different components (chunking, embedding, retrieval, re-ranking) to be swapped and evaluated independently
- Custom embedding models can be integrated to support various vector stores and databases
- The re-ranker stage is optional and can be enabled/disabled based on evaluation needs
