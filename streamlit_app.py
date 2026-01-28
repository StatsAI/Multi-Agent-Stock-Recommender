import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import re
from fpdf import FPDF
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

st.title("🤖 Multi-Agent Quantitative Analysis Board")

with st.sidebar:
    st.header("Control Panel")
    api_key = st.secrets.get("open_ai_api_key", "")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
    
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    
    st.divider()
    st.header("Agent Settings")
    risk_tolerance = st.selectbox("Risk Mandate", ["Conservative", "Balanced", "Aggressive"], index=1)
    
    st.divider()
    st.header("Portfolio Scenario")
    investment_amount = st.number_input("Investment Budget ($)", value=10000, step=1000)

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA UTILITIES ---
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    lt_df = stock.history(period="1y", interval="1d")
    st_df = stock.history(period="5d", interval="5m")
    return lt_df, st_df

def clean_text(text):
    """Replaces Unicode characters that 'helvetica' cannot render."""
    if not text: return ""
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2011": "-",  # Dashes/Hyphens
        "\u2018": "'", "\u2019": "'",                 # Single quotes
        "\u201c": '"', "\u201d": '"',                 # Double quotes
        "\u2022": "*",                                # Bullets
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text.encode("latin-1", "ignore").decode("latin-1")

# --- 3. PDF GENERATION UTILITY ---
class StockReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'Board of Directors: Quantitative Investment Report', 0, 1, 'C')
        self.ln(5)

def generate_report_pdf(ticker, recommendation, reports, risk):
    pdf = StockReportPDF()
    pdf.add_page()
    
    # Title & Executive Summary
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, f"Analysis: {ticker} | Strategy: {risk}", ln=True)
    pdf.ln(5)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", ln=True, fill=True)
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 8, clean_text(recommendation))
    
    # Analyst Tabs
    for name, content in reports.items():
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, f"{name} Analyst Report", ln=True)
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(0, 7, clean_text(content))
        
    # CRITICAL: Return as bytes() to avoid StreamlitAPIException
    return bytes(pdf.output())

# --- 4. LANGGRAPH AGENTS ---
class AgentState(TypedDict):
    ticker: str
    risk: str
    lt_summary: str
    st_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    forecasting_report: str
    news_report: str
    news_sentiment: float
    final_recommendation: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def news_node(state: AgentState):
    prompt = f"Sentiment for {state['ticker']}. End with 'SENTIMENT_SCORE: X' (-1.0 to 1.0)."
    res = llm.invoke(prompt)
    score_match = re.search(r"SENTIMENT_SCORE:\s*([\d\.-]+)", res.content)
    return {"news_report": res.content, "news_sentiment": float(score_match.group(1)) if score_match else 0.0}

# (Other nodes like fundamental_node, supervisor_node etc. follow the same logic as previous)
# ... [Insert Node Definitions here] ...

# --- 5. GRAPH BUILDING ---
builder = StateGraph(AgentState)
# ... [Insert builder.add_node and builder.add_edge logic here] ...
graph = builder.compile()

# --- 6. APP EXECUTION ---
if api_key:
    if st.button("Generate Board Report"):
        with st.spinner("Analyzing..."):
            lt_df, st_df = get_financial_data(selected_ticker)
            
            # Dashboard Head
            c1, c2, c3 = st.columns([1.5, 1.5, 1])
            with c1: st.caption("1Y Trend"); st.line_chart(lt_df['Close'], height=200)
            with c2: st.caption("5D Momentum"); st.line_chart(st_df['Close'], height=200)
            with c3: gauge_placeholder = st.empty()

            final_state = graph.invoke({
                "ticker": selected_ticker, 
                "lt_summary": str(lt_df['Close'].iloc[-1]), 
                "st_summary": str(st_df['Close'].iloc[-1]), 
                "risk": risk_tolerance
            })

            # Rendering
            st.header("Executive Summary")
            st.success(final_state['final_recommendation'])
            
            # PDF Fix
            reports = {"News": final_state['news_report'], "Final": final_state['final_recommendation']}
            pdf_data = generate_report_pdf(selected_ticker, final_state['final_recommendation'], reports, risk_tolerance)
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data, # Now a proper bytes object
                file_name=f"{selected_ticker}_Report.pdf",
                mime="application/pdf"
            )
