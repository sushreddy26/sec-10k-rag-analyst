import os
import glob
import chromadb
from typing import List, Optional
from sec_edgar_downloader import Downloader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from src.config import Config
from src.parser import FilingParser

class MultiCompanyIndexManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self.chroma_collection = self.chroma_client.get_or_create_collection(Config.COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # FastEmbed ONNX runtime for low-latency CPU embedding
        self.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            max_length=512,
            threads=4
        )
        self.downloader = Downloader(Config.USER_AGENT_NAME, Config.USER_AGENT_EMAIL, Config.RAW_DATA_DIR)

    def is_filing_indexed(self, ticker: str, year: str) -> bool:
        results = self.chroma_collection.get(
            where={"$and": [{"ticker": ticker.upper()}, {"year": str(year)}]},
            limit=1
        )
        return len(results.get("ids", [])) > 0

    def _find_primary_10k_file(self, ticker: str) -> Optional[str]:
        filing_dir = os.path.join(Config.RAW_DATA_DIR, "sec-edgar-filings", ticker, "10-K")
        all_candidates = glob.glob(os.path.join(filing_dir, "**", "*.htm*"), recursive=True)
        valid_files = [f for f in all_candidates if not any(x in f.lower() for x in ["filing-details", "xsl", "schema"])]
        if not valid_files:
            valid_files = all_candidates
        return max(valid_files, key=os.path.getsize) if valid_files else None

    def ensure_ticker_year_indexed(self, ticker: str, year: str):
        ticker = ticker.upper().strip()
        year = str(year).strip()

        if self.is_filing_indexed(ticker, year):
            print(f"[✓] {ticker} ({year}) is already indexed in ChromaDB.")
            return

        print(f"[+] Fetching 10-K for {ticker} from SEC EDGAR...")
        next_year = str(int(year) + 1)
        try:
            self.downloader.get(
                "10-K",
                ticker,
                after=f"{year}-01-01",
                before=f"{next_year}-06-30",
                download_details=True
            )
        except Exception as e:
            print(f"[!] Bounded download failed: {e}. Falling back to general retrieval...")

        target_file = self._find_primary_10k_file(ticker)

        if not target_file:
            print(f"[+] Retrying without strict date boundary for {ticker} (limit=2)...")
            self.downloader.get("10-K", ticker, limit=2, download_details=True)
            target_file = self._find_primary_10k_file(ticker)

        if not target_file:
            raise FileNotFoundError(
                f"Could not find or download any 10-K filings for '{ticker}'. "
                f"Verify the ticker symbol is registered with the SEC EDGAR system."
            )

        print(f"[+] Found primary filing: {os.path.basename(target_file)}")
        print(f"[+] Parsing document structure (isolating tables vs narratives) for {ticker} ({year})...")
        
        parser = FilingParser(ticker=ticker, year=year)
        nodes = parser.parse_html(target_file)

        if not nodes:
            raise ValueError(f"Filing parser could not extract textual or tabular nodes from {target_file}")

        for n in nodes:
            n.metadata["ticker"] = ticker
            n.metadata["year"] = year

        print(f"[+] Indexing {len(nodes)} chunks into ChromaDB for {ticker} ({year})...")
        VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        print(f"[✓] Successfully indexed {ticker} ({year}).")

    def retrieve_company_nodes(self, ticker: str, year: str, query: str, top_k: int = Config.TOP_K_PER_COMPANY):
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="ticker", value=ticker.upper().strip()),
                ExactMatchFilter(key="year", value=str(year).strip())
            ]
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        return retriever.retrieve(query)
