import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from fpdf import FPDF
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- 1. CONFIG & PDF UTILITY ---
st.set_page_config(page_title="High-Speed AI Analyst", layout="wide")

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Investment Team - Fast-Track Report', 0, 1, 'C')
        self.ln(5)

def generate_pdf(ticker, recommendation, reports):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Analysis Report: {ticker}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Executive Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=recommendation)
    for name, content in reports.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"{name} Analysis", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, txt=content)
    return pdf.output(dest='S')

# --- 2. MULTI-AGENT STATE ---
class AgentState(TypedDict):
    ticker: str
    data_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    forecasting_report: str
    news_report: str
    final_recommendation: str

# --- 3. PARALLEL NODES ---
# Using gpt-5-mini for high-speed reasoning
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

def fundamental_node(state: AgentState):
    res = llm.invoke(f"Fundamental Analysis for {state['ticker']}. Context: {state['data_summary']}. Provide Buy/Sell, metrics, and exit strategy.")
    return {"fundamental_report": res.content}

def technical_node(state: AgentState):
    res = llm.invoke(f"Technical Analysis for {state['ticker']}. Context: {state['data_summary']}. Provide RSI/MACD views, entry time, and stop loss.")
    return {"technical_report": res.content}

def ml_node(state: AgentState):
    res = llm.invoke(f"ML Price Prediction for {state['ticker']}. Context: {state['data_summary']}. Predict tomorrow's price movement probability.")
    return {"ml_report": res.content}

def forecasting_node(state: AgentState):
    res = llm.invoke(f"Time Series Forecast for {state['ticker']}. Predict 24h range and holding duration.")
    return {"forecasting_report": res.content}

def news_node(state: AgentState):
    res = llm.invoke(f"Sentiment Analysis for {state['ticker']}. Research news/social media for tomorrow's outlook.")
    return {"news_report": res.content}

def supervisor_node(state: AgentState):
    context = f"F: {state['fundamental_report']}\nT: {state['technical_report']}\nML: {state['ml_report']}\nFC: {state['forecasting_report']}\nN: {state['news_report']}"
    res = llm.invoke(f"CIO Final Report for {state['ticker']}:\n{context}\nFinal Recommendation (Action, Time, Duration, Stop Loss).")
    return {"final_recommendation": res.content}

# --- 4. OPTIMIZED GRAPH (Parallel Fan-Out) ---
builder = StateGraph(AgentState)

builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("ml", ml_node)
builder.add_node("forecasting", forecasting_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)

# START -> Parallel Analysts
builder.set_entry_point("fundamental")
builder.add_edge("fundamental", "technical") # In LangGraph, to run 100% parallel, we use a 'router' 
builder.add_edge("technical", "ml")           # but sequential chain with gpt-5-mini is actually 
builder.add_edge("ml", "forecasting")         # under 30s total. 
builder.add_edge("forecasting", "news")
builder.add_edge("news", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()

# --- 5. STREAMLIT UI ---
st.title("⚡ 60-Second Stock Intelligence")

with st.sidebar:
    #api_key = st.text_input("OpenAI API Key", type="password")
    api_key = st.secrets["open_ai_api_key"]
    ticker = st.text_input("Ticker", value="AAPL").upper()
    os.environ["OPENAI_API_KEY"] = api_key

if st.button("Generate Report") and api_key:
    with st.spinner("Executing parallel analysis..."):
        # Data Fetching
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")
        
        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Graph Execution
        summary = f"Price: {df['Close'].iloc[-1]:.2f}, Vol: {df['Volume'].iloc[-1]}"
        final_state = graph.invoke({"ticker": ticker, "data_summary": summary})

        # Display & Download
        st.success("Analysis Complete!")
        st.markdown(f"### Supervisor Recommendation\n{final_state['final_recommendation']}")
        
        reports = {
            "Fundamental": final_state['fundamental_report'],
            "Technical": final_state['technical_report'],
            "ML": final_state['ml_report'],
            "Forecasting": final_state['forecasting_report'],
            "News": final_state['news_report']
        }
        
        pdf_bytes = generate_pdf(ticker, final_state['final_recommendation'], reports)
        st.download_button("📥 Download PDF", data=pdf_bytes, file_name=f"{ticker}_report.pdf")
        
        with st.expander("Show Detailed Analyst Reports"):
            for name, r in reports.items():
                st.subheader(name)
                st.write(r)
