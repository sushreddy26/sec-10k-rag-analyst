import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from llama_index.core.schema import TextNode
from typing import List

# Suppress SEC EDGAR HTML/XML parser warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class FilingParser:
    def __init__(self, ticker: str, year: str):
        self.ticker = str(ticker).upper()
        self.year = str(year)

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def table_to_markdown(self, table_soup) -> str:
        rows = table_soup.find_all("tr")
        if len(rows) < 2:
            return ""
        
        md_rows = []
        for row in rows[:35]:
            cols = [self.clean_text(td.get_text()) for td in row.find_all(["td", "th"])]
            if any(cols):
                md_rows.append("| " + " | ".join(cols) + " |")

        if len(md_rows) > 1:
            header_sep = "| " + " | ".join(["---"] * len(rows[0].find_all(["td", "th"]))) + " |"
            md_rows.insert(1, header_sep)
            return "\n".join(md_rows)
        return ""

    def parse_html(self, html_path: str) -> List[TextNode]:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")

        nodes = []
        for table in soup.find_all("table"):
            table_md = self.table_to_markdown(table)
            if len(table_md) > 60:
                nodes.append(TextNode(
                    text=f"### Financial Table ({self.ticker} {self.year})\n" + table_md,
                    metadata={"ticker": self.ticker, "year": self.year, "chunk_type": "table"}
                ))

        for p in soup.find_all(["p"]):
            text = self.clean_text(p.get_text())
            if len(text) > 120:
                nodes.append(TextNode(
                    text=text,
                    metadata={"ticker": self.ticker, "year": self.year, "chunk_type": "narrative"}
                ))

        if len(nodes) < 5:
            for div in soup.find_all(["div"]):
                text = self.clean_text(div.get_text())
                if 120 < len(text) < 2000:
                    nodes.append(TextNode(
                        text=text,
                        metadata={"ticker": self.ticker, "year": self.year, "chunk_type": "narrative"}
                    ))

        return nodes

    def parse_html_targeted(self, html_path: str) -> List[TextNode]:
        return self.parse_html(html_path)
