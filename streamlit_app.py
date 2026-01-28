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
st.set_page_config(page_title="Investment Committee AI", layout="wide")
st.title("🏦 Institutional Investment Committee")

with st.sidebar:
    api_key = st.secrets.get("open_ai_api_key") or st.text_input("OpenAI API Key", type="password")
    selected_ticker = st.text_input("Ticker", value="NVDA").upper()
    risk_tolerance = st.selectbox("Mandate", ["Conservative", "Balanced", "Aggressive"], index=1)
    investment_amount = st.number_input("Budget ($)", value=10000)
    if api_key: os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA UTILITIES ---
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1y"), stock.history(period="5d", interval="5m")

def clean_text(text):
    if not text: return ""
    rep = {"\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
    for k, v in rep.items(): text = text.replace(k, v)
    return text.encode("latin-1", "ignore").decode("latin-1")

# --- 3. AGENT DEFINITIONS (High-Conviction) ---
class AgentState(TypedDict):
    ticker: str
    risk: str
    lt_summary: str
    st_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    news_report: str
    news_sentiment: float
    final_recommendation: str

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

def fundamental_node(state: AgentState):
    prompt = f"You are a Senior Fundamental Analyst. TASK: Review {state['ticker']} for a {state['risk']} portfolio. Based on {state['lt_summary']}, issue a definitive BUY or SELL rating. No 'on the other hand' hedging. Give 1 price target."
    return {"fundamental_report": llm.invoke(prompt).content}

def technical_node(state: AgentState):
    prompt = f"You are a Quant Technician. TASK: Analyze {state['st_summary']}. Provide a Bullish/Bearish stance. Specify EXACT Entry and Stop Loss levels. No vague ranges."
    return {"technical_report": llm.invoke(prompt).content}

def news_node(state: AgentState):
    prompt = f"Analyze sentiment for {state['ticker']}. State if news is ACCELERATING or DECELERATING price. End with: SENTIMENT_SCORE: X (from -1.0 to 1.0)."
    res = llm.invoke(prompt).content
    score = float(re.search(r"SENTIMENT_SCORE:\s*([\d\.-]+)", res).group(1)) if "SENTIMENT_SCORE" in res else 0.0
    return {"news_report": res, "news_sentiment": score}

def supervisor_node(state: AgentState):
    prompt = f"""
    You are the Chief Investment Officer. Synthesize these reports:
    F: {state['fundamental_report']}
    T: {state['technical_report']}
    N: {state['news_report']}
    
    TASK: ISSUE A BINDING TRADE ORDER. 
    FORMAT: 
    - ACTION: [BUY/SELL/HOLD]
    - ALLOCATION: [Percentage of {state['risk']} budget]
    - ENTRY/EXIT: [Specific Prices]
    - RATIONALE: [One sentence of core logic]
    """
    return {"final_recommendation": llm.invoke(prompt).content}

# --- 4. GRAPH CONSTRUCTION (The Entrypoint Fix) ---
builder = StateGraph(AgentState)
builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)

# Connect START to initial parallel nodes
builder.add_edge(START, "fundamental")
builder.add_edge(START, "technical")
builder.add_edge(START, "news")

# Connect parallel nodes to supervisor
builder.add_edge("fundamental", "supervisor")
builder.add_edge("technical", "supervisor")
builder.add_edge("news", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()

# --- 5. PDF GENERATION ---
def generate_report_pdf(ticker, recommendation, reports):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, clean_text(f"INVESTMENT COMMITTEE ORDER: {ticker}"), ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 8, clean_text(recommendation))
    return bytes(pdf.output())

# --- 6. EXECUTION ---
if api_key and st.button("EXECUTE TRADE DELIBERATION"):
    with st.spinner("Committee in session..."):
        lt_df, st_df = get_financial_data(selected_ticker)
        lt_s = f"Price: {lt_df['Close'].iloc[-1]:.2f}, 1Y Chg: {((lt_df['Close'].iloc[-1]/lt_df['Close'].iloc[0])-1)*100:.1f}%"
        st_s = f"5D Vol: {st_df['Close'].std():.2f}"
        
        result = graph.invoke({"ticker": selected_ticker, "lt_summary": lt_s, "st_summary": st_s, "risk": risk_tolerance})
        
        st.header("🏁 Final Committee Decision")
        st.code(result['final_recommendation'], language="markdown")
        
        pdf_data = generate_report_pdf(selected_ticker, result['final_recommendation'], {})
        st.download_button("📥 Download Official Trade Ticket", data=pdf_data, file_name="Trade_Order.pdf")
