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
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze P/E, Debt/Equity, and Cash Flow. State a definitive Buy/Hold/Sell position. Do not ask for guidance; use your expertise to explain exactly how these metrics dictate your 1-day outlook."
    res = llm.invoke(prompt)
    return {"fundamental_report": res.content}

def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze RSI, MACD, and Moving Averages. Provide a concrete entry time for tomorrow and a specific exit price. Explain the logic behind these levels based on current support and resistance."
    res = llm.invoke(prompt)
    return {"technical_report": res.content}

def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}. Using historical price patterns in {state['data_summary']}, dictate the probability of a price increase tomorrow. State your confidence level and specific price targets decisively based on the pattern recognition models you've run."
    res = llm.invoke(prompt)
    return {"ml_report": res.content}

def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecasting Analyst. Analyze the trend for {state['ticker']}. Provide a specific price range for the next 24-48 hours and a fixed holding duration. Explain how the trend momentum led you to this specific forecast."
    res = llm.invoke(prompt)
    return {"forecasting_report": res.content}

def news_node(state: AgentState):
    prompt = f"Act as a Sentiment & News Analyst for {state['ticker']}. Scan for recent news and macro events. Decisively state how tomorrow's market will react to these events. Do not provide 'if/then' scenarios; provide your best judgment on the net sentiment impact."
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
    prompt = f"You are the Chief Investment Officer. Synthesize the expert reports for {state['ticker']} and issue a FINAL EXECUTIVE COMMAND. You must provide: 1. Action (BUY/SELL/WAIT), 2. Precise Purchase/Sale Time, 3. Exact Holding Duration, 4. Strict Stop Loss Price. Base this on the consensus or weight of evidence from your team."
    res = llm.invoke(prompt + context)
    return {"final_recommendation": res.content}

# --- 5. GRAPH CONSTRUCTION (PARALLEL EXECUTION) ---
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("ml", ml_node)
builder.add_node("forecasting", forecasting_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)

# Set up Parallel entry: All analysts start at the same time
builder.set_entry_point("fundamental")
# Note: LangGraph executes independent nodes in parallel automatically 
# when they share the same starting point or are reached simultaneously.
builder.add_edge("__start__", "technical")
builder.add_edge("__start__", "ml")
builder.add_edge("__start__", "forecasting")
builder.add_edge("__start__", "news")

# All analysts converge on the supervisor
builder.add_edge("fundamental", "supervisor")
builder.add_edge("technical", "supervisor")
builder.add_edge("ml", "supervisor")
builder.add_edge("forecasting", "supervisor")
builder.add_edge("news", "supervisor")

builder.add_edge("supervisor", END)
graph = builder.compile()

# --- 6. APP EXECUTION ---
if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start analysis.")
else:
    if st.button(f"Analyze {selected_ticker}"):
        with st.spinner("Executing parallel analysis and convening the agent team..."):
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
