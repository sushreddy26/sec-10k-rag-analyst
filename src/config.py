import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    USER_AGENT_NAME: str = "AcademicResearcher"
    USER_AGENT_EMAIL: str = os.getenv("SEC_EDGAR_USER_AGENT", "yourname@gmail.com")
    CHROMA_PERSIST_DIR: str = "./vector_store"
    RAW_DATA_DIR: str = "./data/raw_filings"
    COLLECTION_NAME: str = "sec_10k_multi_company"
    
    # Fast, local CPU-friendly embedding model
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Lightweight local LLM (Runs smoothly on macOS CPU/MPS without 16GB memory stall)
    LLM_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Fast, highly accurate at financial reasoning
    
    TOP_K_PER_COMPANY: int = 3
 