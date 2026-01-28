import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from fpdf import FPDF
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

st.title("🤖 Multi-Agent Quantitative Analysis Team")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    # API Key from Streamlit Secrets
    api_key = st.secrets.get("open_ai_api_key", "")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
    
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    timeframe = st.selectbox("Chart Timeframe", ["1mo", "6mo", "1y"], index=0)
    
    st.divider()
    st.header("Agent Settings")
    risk_tolerance = st.selectbox(
        "Risk Tolerance", 
        ["Conservative", "Balanced", "Aggressive"], 
        index=1,
        help="Adjusts how likely agents are to recommend a purchase based on volatility."
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA UTILITIES ---
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval="1d")
    intraday = stock.history(period="1d", interval="5m")
    info = stock.info
    return df, intraday, info

# --- 3. PDF GENERATION UTILITY ---
class StockReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'Quantitative Analysis Board Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_report_pdf(ticker, recommendation, reports, risk):
    pdf = StockReportPDF()
    pdf.add_page()
    
    # Title Section
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, f"Analysis: {ticker} ({risk} Strategy)", ln=True)
    pdf.ln(5)
    
    # Executive Summary
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", ln=True, fill=True)
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 8, recommendation)
    pdf.ln(10)
    
    # Analyst Details
    for name, content in reports.items():
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, f"{name} Analyst Deep Dive", ln=True)
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(0, 7, content)
        pdf.ln(5)
        
    return pdf.output()

# --- 4. LANGGRAPH AGENT STATE ---
class AgentState(TypedDict):
    ticker: str
    risk: str
    data_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    forecasting_report: str
    news_report: str
    final_recommendation: str

# --- 5. AGENT NODES ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

def fundamental_node(state: AgentState):
    prompt = f"Act as a Fundamental Analyst for {state['ticker']} with a {state['risk']} risk profile. Data: {state['data_summary']}. Analyze P/E, Debt/Equity, and Cash Flow. Provide a 1-day outlook with concrete metrics."
    res = llm.invoke(prompt)
    return {"fundamental_report": res.content}

def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']} ({state['risk']} risk). Data: {state['data_summary']}. Analyze RSI, MACD, and MAs. Recommend an entry time and exit strategy."
    res = llm.invoke(prompt)
    return {"technical_report": res.content}

def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']} ({state['risk']} risk). Context: {state['data_summary']}. Predict price increase probability and targets."
    res = llm.invoke(prompt)
    return {"ml_report": res.content}

def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecasting Analyst for {state['ticker']} ({state['risk']} risk). Predict 24-48 hour price range and holding duration."
    res = llm.invoke(prompt)
    return
