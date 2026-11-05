import os
from sec_edgar_downloader import Downloader
from src.config import Config

class EdgarDownloader:
    def __init__(self, download_dir: str = Config.RAW_DATA_DIR):
        email = Config.USER_AGENT_EMAIL
        company_name = email.split()[0] if " " in email else "ResearchLab"
        user_agent_str = f"{company_name} {email}"
        self.dl = Downloader(company_name, email, download_dir)

    def fetch_10k(self, ticker: str, limit: int = 1):
        """Downloads the latest 10-K filing for a given ticker."""
        self.dl.get("10-K", ticker, limit=limit, download_details=True)
        print(f"[+] Download complete for {ticker} (Limit: {limit})")

if __name__ == "__main__":
    downloader = EdgarDownloader()
    downloader.fetch_10k("AAPL", limit=1)