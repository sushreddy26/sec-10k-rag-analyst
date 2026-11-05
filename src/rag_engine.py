import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from src.config import Config
from src.indexer import MultiCompanyIndexManager

def get_lightweight_llm():
    if torch.backends.mps.is_available():
        target_device = "mps"
        model_dtype = torch.float16
        print("[+] Hardware acceleration: Using Apple Silicon Metal (MPS)")
    elif torch.cuda.is_available():
        target_device = "cuda"
        model_dtype = torch.float16
        print("[+] Hardware acceleration: Using NVIDIA CUDA")
    else:
        target_device = "cpu"
        model_dtype = torch.float32
        print("[+] Hardware acceleration: Falling back to CPU")

    return HuggingFaceLLM(
        model_name=Config.LLM_MODEL_ID,
        tokenizer_name=Config.LLM_MODEL_ID,
        context_window=4096,
        max_new_tokens=768,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        device_map=target_device,
        model_kwargs={"torch_dtype": model_dtype}
    )

COMPARISON_PROMPT = """You are an expert financial analyst. Analyze and compare the following two filings strictly based on the extracted 10-K sections below.

==================================================
COMPANY A: {ticker_a} (Fiscal Year {year_a})
{context_a}
==================================================
COMPANY B: {ticker_b} (Fiscal Year {year_b})
{context_b}
==================================================

QUESTION / USER GOAL:
{query}

Format your output with these exact sections:
### 1. Direct Metric Comparison (Tables & Key Numbers)
- Clearly state the figures for {ticker_a} ({year_a}) vs. {ticker_b} ({year_b}). Do NOT mix them up.
- Indicate if each figure came from a financial statement table or management narrative.

### 2. What This Means in Plain English
- Explain the significance of these numbers to an everyday reader without financial jargon.

### 3. Summary & Takeaways
- Provide a 2-3 sentence wrap-up highlighting the key strategic difference.
"""

class ComparativeFinancialEngine:
    def __init__(self):
        print("[+] Initializing MultiCompanyIndexManager...")
        self.index_mgr = MultiCompanyIndexManager()
        print("[+] Loading local LLM engine...")
        self.llm = get_lightweight_llm()

    def stream_compare_companies(self, ticker_a: str, year_a: str, ticker_b: str, year_b: str, query: str):
        ticker_a = ticker_a.upper().strip()
        ticker_b = ticker_b.upper().strip()
        year_a = str(year_a).strip()
        year_b = str(year_b).strip()

        self.index_mgr.ensure_ticker_year_indexed(ticker_a, year_a)
        self.index_mgr.ensure_ticker_year_indexed(ticker_b, year_b)

        nodes_a = self.index_mgr.retrieve_company_nodes(ticker_a, year_a, query)
        nodes_b = self.index_mgr.retrieve_company_nodes(ticker_b, year_b, query)

        context_a = "\n\n".join([f"[{n.metadata.get('chunk_type', 'text').upper()}] {n.text}" for n in nodes_a])
        context_b = "\n\n".join([f"[{n.metadata.get('chunk_type', 'text').upper()}] {n.text}" for n in nodes_b])

        prompt = COMPARISON_PROMPT.format(
            ticker_a=ticker_a,
            year_a=year_a,
            ticker_b=ticker_b,
            year_b=year_b,
            context_a=context_a if context_a else "No direct data found.",
            context_b=context_b if context_b else "No direct data found.",
            query=query
        )

        citations = {
            f"{ticker_a} ({year_a})": [{"score": f"{n.score:.4f}" if n.score else "N/A", "text": n.text[:200] + "..."} for n in nodes_a],
            f"{ticker_b} ({year_b})": [{"score": f"{n.score:.4f}" if n.score else "N/A", "text": n.text[:200] + "..."} for n in nodes_b]
        }

        # Live token stream generator
        token_stream = self.llm.stream_complete(prompt)
        return token_stream, citations

    def compare_companies(self, ticker_a: str, year_a: str, ticker_b: str, year_b: str, query: str):
        token_stream, citations = self.stream_compare_companies(ticker_a, year_a, ticker_b, year_b, query)
        full_text = "".join([chunk.delta for chunk in token_stream])
        return full_text, citations
