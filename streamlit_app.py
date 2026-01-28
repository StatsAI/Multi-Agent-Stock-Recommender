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
    pdf.cell(0, 10, f"Analysis: {ticker} | Strategy: {risk}", ln=True)
    pdf.ln(5)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", ln=True, fill=True)
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 8, recommendation)
    for name, content in reports.items():
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, f"{name} Analyst Report", ln=True)
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(0, 7, content)
    return pdf.output()

# --- 4. LANGGRAPH AGENT STATE ---
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

# --- 5. PARALLEL AGENT NODES ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

def fundamental_node(state: AgentState):
    res = llm.invoke(f"Fundamental Analysis for {state['ticker']} ({state['risk']} risk). Context: {state['lt_summary']}.")
    return {"fundamental_report": res.content}

def technical_node(state: AgentState):
    res = llm.invoke(f"Technical Analysis for {state['ticker']} ({state['risk']} risk). Momentum: {state['st_summary']}.")
    return {"technical_report": res.content}

def ml_node(state: AgentState):
    res = llm.invoke(f"ML Prediction for {state['ticker']}. Target price based on: {state['st_summary']}.")
    return {"ml_report": res.content}

def forecasting_node(state: AgentState):
    res = llm.invoke(f"Forecasting for {state['ticker']}. Duration & range using: {state['st_summary']}.")
    return {"forecasting_report": res.content}

def news_node(state: AgentState):
    prompt = f"News Sentiment for {state['ticker']}. Based on macro news, provide report. End with 'SENTIMENT_SCORE: X' (-1.0 to 1.0)."
    res = llm.invoke(prompt)
    score_match = re.search(r"SENTIMENT_SCORE:\s*([\d\.-]+)", res.content)
    score = float(score_match.group(1)) if score_match else 0.0
    return {"news_report": res.content, "news_sentiment": score}

def supervisor_node(state: AgentState):
    reports = f"F: {state['fundamental_report']}\nT: {state['technical_report']}\nML: {state['ml_report']}\nFC: {state['forecasting_report']}\nN: {state['news_report']}"
    res = llm.invoke(f"Review for {state['ticker']} ({state['risk']} risk). Final Buy/Sell, Entry, Duration, Stop Loss.\n\n{reports}")
    return {"final_recommendation": res.content}

# --- 6. GRAPH CONSTRUCTION ---
builder = StateGraph(AgentState)
builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("ml", ml_node)
builder.add_node("forecasting", forecasting_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)

builder.add_edge(START, "fundamental")
builder.add_edge(START, "technical")
builder.add_edge(START, "ml")
builder.add_edge(START, "forecasting")
builder.add_edge(START, "news")

builder.add_edge("fundamental", "supervisor")
builder.add_edge("technical", "supervisor")
builder.add_edge("ml", "supervisor")
builder.add_edge("forecasting", "supervisor")
builder.add_edge("news", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()

# --- 7. APP EXECUTION ---
if api_key:
    if st.button(f"Generate High-Speed Board Report"):
        with st.spinner("Parallelizing analyst workflows..."):
            lt_df, st_df, info = get_financial_data(selected_ticker)
            
            # Dashboard Head
            c1, c2, c3 = st.columns([1.5, 1.5, 1])
            with c1: st.caption("Long-Term Trend"); st.line_chart(lt_df['Close'], height=200)
            with c2: st.caption("Intra-day Momentum"); st.line_chart(st_df['Close'], height=200)
            with c3: gauge_placeholder = st.empty()

            

            # Run Analysis
            lt_sum = f"1Y High: {lt_df['High'].max():.2f}, Trend: {'Up' if lt_df['Close'].iloc[-1] > lt_df['Close'].iloc[0] else 'Down'}"
            st_sum = f"Price: {st_df['Close'].iloc[-1]:.2f}, 5D StdDev: {st_df['Close'].std():.2f}"
            
            final_state = graph.invoke({"ticker": selected_ticker, "lt_summary": lt_sum, "st_summary": st_sum, "risk": risk_tolerance})

            # Gauge Rendering
            fig = go.Figure(go.Indicator(mode="gauge+number", value=final_state['news_sentiment'], 
                                        domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Sentiment"},
                                        gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "white"},
                                               'steps': [{'range': [-1, -0.3], 'color': "red"}, {'range': [0.3, 1], 'color': "green"}]}))
            fig.update_layout(height=250, margin=dict(t=50, b=0), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            gauge_placeholder.plotly_chart(fig, use_container_width=True)

            # Executive Results
            st.header("Executive Summary")
            st.success(final_state['final_recommendation'])
            
            # Scenario Tool
            shares = int(investment_amount / st_df['Close'].iloc[-1])
            st.info(f"💡 At current prices, your ${investment_amount:,} budget allows for a position of **{shares} shares**.")

            # PDF Download
            reports = {"Fundamental": final_state['fundamental_report'], "Technical": final_state['technical_report'], "ML": final_state['ml_report'], "Forecast": final_state['forecasting_report'], "News": final_state['news_report']}
            pdf_bytes = generate_report_pdf(selected_ticker, final_state['final_recommendation'], reports, risk_tolerance)
            st.download_button("📥 Download Official Board PDF", data=bytes(pdf_bytes), file_name=f"{selected_ticker}_Board_Report.pdf")

            st.divider()
            tabs = st.tabs(list(reports.keys()))
            for i, report in enumerate(reports.values()):
                with tabs[i]: st.write(report)
