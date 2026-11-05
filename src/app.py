import gradio as gr
from src.rag_engine import ComparativeFinancialEngine

engine = None

def get_engine():
    global engine
    if engine is None:
        print("[+] Initializing ComparativeFinancialEngine...")
        engine = ComparativeFinancialEngine()
    return engine

def process_comparison(ticker_a, year_a, ticker_b, year_b, query, progress=gr.Progress()):
    if not ticker_a.strip():
        yield "⚠️ Error: Please specify at least Company A ticker.", ""
        return

    actual_ticker_b = ticker_b.strip() if ticker_b.strip() else ticker_a.strip()
    actual_year_b = year_b.strip() if year_b.strip() else year_a.strip()

    try:
        progress(0.15, desc=f"Checking filings for {ticker_a.upper()} and {actual_ticker_b.upper()}...")
        rag = get_engine()

        progress(0.40, desc="Retrieving nodes and preparing model context...")
        stream_response, citations = rag.stream_compare_companies(
            ticker_a=ticker_a,
            year_a=year_a,
            ticker_b=actual_ticker_b,
            year_b=actual_year_b,
            query=query
        )

        citation_md = "### Source 10-K Citations\n\n"
        for label, cite_list in citations.items():
            citation_md += f"#### {label}:\n"
            for i, c in enumerate(cite_list, 1):
                citation_md += f"- **Citation [{i}]** (Relevance: {c['score']}):\n  > *{c['text']}*\n\n"

        accumulated_text = ""
        for chunk in stream_response:
            accumulated_text += chunk.delta
            yield accumulated_text, citation_md

    except Exception as exc:
        err_msg = f"### ⚠️ Execution Error\n\n**Details:** {str(exc)}\n\n*Suggestions:*\n- Check if the ticker exists on SEC EDGAR.\n- Ensure the selected fiscal year has an available 10-K."
        yield err_msg, ""

with gr.Blocks(title="SEC 10-K Multi-Company Intelligence Platform") as demo:
    gr.Markdown("# 📊 SEC 10-K Financial Intelligence & Multi-Year Comparator")
    gr.Markdown(
        "Download filings on-demand by **ticker** and **fiscal year**. "
        "Separates tables from narratives to preserve numeric fidelity, isolates metadata to prevent mixups, "
        "and translates complex filings into plain English."
    )

    with gr.Row():
        with gr.Column():
            ticker_a = gr.Textbox(label="Company A Ticker", value="BLK")
            year_a = gr.Dropdown(label="Company A Fiscal Year", choices=["2023", "2022", "2021", "2020"], value="2023")
        with gr.Column():
            ticker_b = gr.Textbox(label="Company B Ticker", value="JPM")
            year_b = gr.Dropdown(label="Company B Fiscal Year", choices=["2023", "2022", "2021", "2020"], value="2023")

    query_box = gr.Textbox(
        label="Comparison Question / Financial Metric",
        value="Compare their total revenue, assets under management (AUM) or total assets, and primary risk factors.",
        lines=2
    )

    submit_btn = gr.Button("Analyze & Compare 10-Ks", variant="primary")

    with gr.Row():
        output_markdown = gr.Markdown(label="Financial Analysis")
        citations_markdown = gr.Markdown(label="Document Citations")

    submit_btn.click(
        fn=process_comparison,
        inputs=[ticker_a, year_a, ticker_b, year_b, query_box],
        outputs=[output_markdown, citations_markdown]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)
