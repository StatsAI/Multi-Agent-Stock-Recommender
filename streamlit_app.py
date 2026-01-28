import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

st.title("🤖 Multi-Agent Quantitative Analysis Team")
st.markdown("""
This app deploys a team of specialized AI agents to analyze stocks using **gpt-5-mini**. 
The team provides a unified recommendation for tomorrow's trading session.
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    #api_key = st.text_input("OpenAI API Key", type="password")
    api_key = st.secrets["open_ai_api_key"]
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    timeframe = st.selectbox("Chart Timeframe", ["1mo", "6mo", "1y"], index=0)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA UTILITIES ---
def get_financial_data(ticker):
    """Fetches comprehensive data for the agents."""
    stock = yf.Ticker(ticker)
    # Get daily data for the chart and analysis
    df = stock.history(period="1mo", interval="1d")
    # Get 1-day intra-day data for precise entry/exit
    intraday = stock.history(period="1d", interval="5m")
    info = stock.info
    return df, intraday, info

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

# --- 4. AGENT NODES ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

def fundamental_node(state: AgentState):
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze P/E, Debt/Equity, and Cash Flow. Provide a Buy/Hold/Sell opinion with concrete metrics and a 1-day outlook."
    res = llm.invoke(prompt)
    return {"fundamental_report": res.content}

def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze RSI, MACD, and Moving Averages. Recommend an entry time for tomorrow and an exit strategy to minimize loss."
    res = llm.invoke(prompt)
    return {"technical_report": res.content}

def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}. Using historical price patterns in {state['data_summary']}, predict the probability of a price increase tomorrow. State your confidence level and specific price targets."
    res = llm.invoke(prompt)
    return {"ml_report": res.content}

def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecasting Analyst. Analyze the trend for {state['ticker']}. Predict the price range for the next 24-48 hours. Suggest a holding duration."
    res = llm.invoke(prompt)
    return {"forecasting_report": res.content}

def news_node(state: AgentState):
    prompt = f"Act as a Sentiment & News Analyst for {state['ticker']}. Scan (simulated) for recent news and social media sentiment. How do macro events impact this stock tomorrow?"
    res = llm.invoke(prompt)
    return {"news_report": res.content}

def supervisor_node(state: AgentState):
    context = f"""
    Fundamental: {state['fundamental_report']}
    Technical: {state['technical_report']}
    ML: {state['ml_report']}
    Forecast: {state['forecasting_report']}
    News: {state['news_report']}
    """
    prompt = f"You are the Chief Investment Officer. Based on the reports below for {state['ticker']}, provide a FINAL RECOMMENDATION. Include: 1. Action (BUY/SELL/WAIT), 2. Purchase Time, 3. Holding Duration, 4. Stop Loss Price."
    res = llm.invoke(prompt + context)
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
builder.add_edge("fundamental", "technical")
builder.add_edge("technical", "ml")
builder.add_edge("ml", "forecasting")
builder.add_edge("forecasting", "news")
builder.add_edge("news", "supervisor")
builder.add_edge("supervisor", END)
graph = builder.compile()

# --- 6. APP EXECUTION ---
if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start analysis.")
else:
    if st.button(f"Analyze {selected_ticker}"):
        with st.spinner("Fetching market data and convening the agent team..."):
            # Fetch Data
            df, intraday, info = get_financial_data(selected_ticker)
            
            # Show Visuals
            st.subheader(f"Market Overview: {selected_ticker}")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
            )])
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            

            # Prepare state context
            summary = f"Recent Close: {df['Close'].iloc[-1]:.2f}, High: {df['High'].max():.2f}, Low: {df['Low'].min():.2f}. Market Cap: {info.get('marketCap')}"
            
            # Run Agents
            initial_state = {"ticker": selected_ticker, "data_summary": summary}
            final_state = graph.invoke(initial_state)

            # Display Results
            st.header("Executive Summary")
            st.success(final_state['final_recommendation'])

            st.divider()
            
            # Display Analyst Details in Tabs
            t1, t2, t3, t4, t5 = st.tabs(["Fundamental", "Technical", "ML", "Forecasting", "News"])
            with t1: st.write(final_state['fundamental_report'])
            with t2: st.write(final_state['technical_report'])
            with t3: st.write(final_state['ml_report'])
            with t4: st.write(final_state['forecasting_report'])
            with t5: st.write(final_state['news_report'])
