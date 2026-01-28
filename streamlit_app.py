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
    risk_tolerance = st.selectbox(
        "Risk Mandate", 
        ["Conservative", "Balanced", "Aggressive"], 
        index=1
    )
    
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
    info = stock.info
    return lt_df, st_df, info

# --- 3. PDF GENERATION UTILITY ---
class StockReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'Board of Directors: Quantitative Investment Report', 0, 1, 'C')
        self.ln(5)

def generate_report_pdf(ticker, recommendation, reports, risk):
    pdf = StockReportPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, f"Analysis: {ticker} | Strategy: {risk}",
