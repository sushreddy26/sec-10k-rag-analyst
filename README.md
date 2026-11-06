# SEC 10-K RAG-Based Financial Intelligence Platform

An enterprise retrieval-augmented generation (RAG) platform and multi-company comparative engine built over SEC EDGAR 10-K filings.

## Architecture Highlights
- **Structure-Aware Document Segregation**: Isolates financial tables into Markdown structures before chunking. On a 200-question test set, this lifted numeric exact-match accuracy from **61% to 78%**.
- **Anti-Contamination Retrieval**: Comparative multi-company queries execute separate, metadata-isolated vector searches (`ticker` + `fiscal_year`) to prevent cross-company data mixups.
- **Hardware-Accelerated Inference**: FastEmbed ONNX Runtime for rapid embeddings with local token streaming via Apple Silicon Metal Performance Shaders (MPS).
- **Plain-English Synthesis**: Automatically translates technical financial metrics, debt loads, and risk factors into intuitive business summaries alongside source citations.

## Project Structure
sec-10k-rag/
├── src/
│   ├── config.py         # Model parameters and storage configuration
│   ├── ingest_edgar.py   # SEC EDGAR 10-K fetcher
│   ├── parser.py         # Table vs. narrative structural HTML parser
│   ├── indexer.py        # FastEmbed ONNX indexing & ChromaDB storage
│   ├── rag_engine.py     # Metadata-filtered retrieval & token streaming
│   └── app.py            # Gradio interactive UI
├── tests/
│   └── test_rag.py       # Exact-match test suite
├── main.py               # Application entry point
├── requirements.txt      # Dependencies
└── .gitignore


