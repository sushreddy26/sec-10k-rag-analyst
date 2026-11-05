import glob
from src.ingest_edgar import EdgarDownloader
from src.parser import FilingParser
from src.indexer import IndexManager
from src.app import demo

def run():
    ticker = "AAPL"
    year = "2023"

    print("==================================================")
    print("STEP 1: Downloading 10-K filing from SEC EDGAR...")
    print("==================================================")
    downloader = EdgarDownloader()
    downloader.fetch_10k(ticker, limit=1)

    print("\n==================================================")
    print("STEP 2: Parsing narrative vs. tables & indexing...")
    print("==================================================")
    parser = FilingParser(ticker=ticker, year=year)
    files = glob.glob(f"data/raw_filings/sec-edgar-filings/{ticker}/10-K/**/*.html", recursive=True)
    if not files:
        files = glob.glob("data/raw_filings/**/*.html", recursive=True)

    if not files:
        raise FileNotFoundError("Could not find downloaded 10-K filing in data/raw_filings.")

    nodes = parser.parse_html(files[0])
    print(f"Extracted {len(nodes)} distinct nodes (segregated tables + text).")

    manager = IndexManager()
    manager.build_and_save_index(nodes)
    print("[+] ChromaDB index built and saved successfully.")

    print("\n==================================================")
    print("STEP 3: Launching Gradio Web UI...")
    print("==================================================")
    demo.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    run()
