import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import asyncio
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

st.title("🤖 Multi-Agent Quantitative Analysis Team")
st.markdown("""
This app deploys a team of specialized AI agents. 
The team analyzes **Short, Medium, and Long-term horizons** to provide a weighted recommendation.
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.secrets["open_ai_api_key"]
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA UTILITIES ---
def get_financial_data(ticker):
    """Fetches comprehensive multi-horizon data for the agents."""
    stock = yf.Ticker(ticker)
    
    # 5-Day Granular Data (Short-term)
    df_5d = stock.history(period="5d", interval="5m")
    # 1-Month Daily Data (Medium-term)
    df_1m = stock.history(period="1mo", interval="1d")
    # 1-Year Daily Data (Long-term)
    df_1y = stock.history(period="1y", interval="1d")
    
    # Fundamental specific data
    info = stock.info
    dividends = stock.dividends
    
    return df_5d, df_1m, df_1y, info, dividends

# --- 3. LANGGRAPH AGENT STATE ---
class AgentState(TypedDict):
    ticker: str
    data_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    forecasting_report: str
    news_report: str
    final_recommendation: str

# --- 4. ASYNC AGENT NODES ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

async def fundamental_node(state: AgentState):
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Data context: {state['data_summary']}. Use the provided dividend history and info to analyze yield, P/E, and Debt/Equity over the past year. State a definitive Buy/Hold/Sell position considering the company's long-term health. Assign a Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"fundamental_report": res.content}

async def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze the short-term (5m) and medium-term (daily) price action. Calculate virtual RSI/MACD based on the high/low/close data provided. Provide a concrete entry time and exit price. Assign a Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"technical_report": res.content}

async def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}. Evaluate price patterns across 5-day, 1-month, and 1-year horizons in the provided data summary. State the probability of a price increase tomorrow vs next week. Assign a Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"ml_report": res.content}

async def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecasting Analyst. Look at the 1-year trend vs the 5-day volatility for {state['ticker']}. Provide a specific price range for the next 48 hours and a forecast for the next 30 days. Assign a Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"forecasting_report": res.content}

async def news_node(state: AgentState):
    prompt = f"Act as a Sentiment Analyst for {state['ticker']}. Evaluate how recent macro events impact both the immediate 24-hour window and the 1-year outlook. State a definitive directional bias. Assign a Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"news_report": res.content}

async def supervisor_node(state: AgentState):
    context = f"Reports: {state['fundamental_report']} | {state['technical_report']} | {state['ml_report']} | {state['forecasting_report']} | {state['news_report']}"
    prompt = f"As CIO, weigh these reports based on confidence and time horizons. Issue a FINAL EXECUTIVE COMMAND for {state['ticker']}: 1. Action, 2. Timing, 3. Duration, 4. Stop Loss."
    res = await llm.ainvoke(prompt + context)
    return {"final_recommendation": res.content}

# --- 5. GRAPH CONSTRUCTION ---
builder = StateGraph(AgentState)
builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("ml", ml_node)
builder.add_node("forecasting", forecasting_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)

builder.set_entry_point("fundamental")
builder.add_edge("__start__", "technical")
builder.add_edge("__start__", "ml")
builder.add_edge("__start__", "forecasting")
builder.add_edge("__start__", "news")
builder.add_edge("fundamental", "supervisor")
builder.add_edge("technical", "supervisor")
builder.add_edge("ml", "supervisor")
builder.add_edge("forecasting", "supervisor")
builder.add_edge("news", "supervisor")
builder.add_edge("supervisor", END)
graph = builder.compile()

# --- 6. APP EXECUTION ---
if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
else:
    if st.button(f"Analyze {selected_ticker}"):
        with st.spinner("Analyzing Short, Medium, and Long-term horizons..."):
            df_5d, df_1m, df_1y, info, dividends = get_financial_data(selected_ticker)
            
            # Charts Displayed Vertically
            st.caption("5-Day Candle (Short)")
            fig1 = go.Figure(data=[go.Candlestick(x=df_5d.index, open=df_5d['Open'], high=df_5d['High'], low=df_5d['Low'], close=df_5d['Close'])])
            fig1.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig1, use_container_width=True)

            st.caption("1-Month Candle (Medium)")
            fig2 = go.Figure(data=[go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close'])])
            fig2.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig2, use_container_width=True)

            st.caption("1-Year Line (Long)")
            fig3 = go.Figure(data=[go.Scatter(x=df_1y.index, y=df_1y['Close'], mode='lines', line=dict(color='cyan'))])
            fig3.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig3, use_container_width=True)

            # Data Preparation for Agents
            div_summary = dividends.tail(5).to_string() if not dividends.empty else "No dividends"
            summary = (f"TICKER: {selected_ticker}\n"
                       f"INFO: P/E: {info.get('trailingPE')}, Mkt Cap: {info.get('marketCap')}, Div Yield: {info.get('dividendYield')}\n"
                       f"DIVIDENDS (Recent): {div_summary}\n"
                       f"5D CLOSE: {df_5d['Close'].iloc[-1]:.2f}\n"
                       f"1M RANGE: {df_1m['Low'].min():.2f} - {df_1m['High'].max():.2f}\n"
                       f"1Y TREND: Start {df_1y['Close'].iloc[0]:.2f} to End {df_1y['Close'].iloc[-1]:.2f}")

            initial_state = {"ticker": selected_ticker, "data_summary": summary}
            final_state = asyncio.run(graph.ainvoke(initial_state))

            st.header("Executive Summary")
            st.success(final_state['final_recommendation'])
            
            st.divider()
            tabs = st.tabs(["Fundamental", "Technical", "ML", "Forecasting", "News"])
            reports = ['fundamental_report', 'technical_report', 'ml_report', 'forecasting_report', 'news_report']
            for i, tab in enumerate(tabs):
                with tab: st.write(final_state[reports[i]])
