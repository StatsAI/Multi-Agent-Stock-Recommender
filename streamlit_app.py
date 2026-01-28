import streamlit as st
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from PIL import Image
from fpdf import FPDF

# --- PDF UTILITIES ---
def clean_text_for_pdf(text):
    """Replaces common unicode characters that cause Helvetica to crash."""
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2022': '*',  # bullet point
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_pdf(ticker, final_state):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    effective_width = 210 - 30  # A4 width minus margins

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(effective_width, 15, f"AI Wall Street Report: {ticker}", ln=True, align='C')
    pdf.set_draw_color(50, 50, 50)
    pdf.line(15, 25, 195, 25)
    pdf.ln(10)

    def write_formatted_content(text):
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue

            pdf.set_x(15)

            if line.startswith('###'):
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(effective_width, 6, line.replace('###', '').strip())
                pdf.set_font("Helvetica", "", 10)

            elif line.startswith('* ') or line.startswith('- '):
                pdf.set_font("Helvetica", "", 10)
                bullet_text = f"- {line[2:].strip()}"
                pdf.multi_cell(effective_width, 5, bullet_text)

            else:
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(effective_width, 5, line)

    # Executive Recommendation
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_x(15)
    pdf.cell(effective_width, 10, " EXECUTIVE MULTI-HORIZON RECOMMENDATION", ln=True, fill=True)
    pdf.ln(2)
    write_formatted_content(clean_text_for_pdf(final_state['final_recommendation']))
    pdf.ln(10)

    reports = {
        "Fundamental Analysis": 'fundamental_report',
        "Technical Analysis": 'technical_report',
        "ML Analysis": 'ml_report',
        "Forecasting Analysis": 'forecasting_report',
        "News & Sentiment": 'news_report'
    }

    for title, key in reports.items():
        if pdf.get_y() > 260:
            pdf.add_page()

        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(15)
        pdf.cell(effective_width, 10, title.upper(), ln=True)

        content = clean_text_for_pdf(final_state[key])
        write_formatted_content(content)
        pdf.ln(8)

    return bytes(pdf.output())

# --- Rest of the Streamlit App (Unchanged) ---
logo = Image.open('picture.png')

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: -60px;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image(logo)

st.markdown("""
        <style>
                .block-container {
                padding-top: 0;
                }
        </style>
        """, unsafe_allow_html=True)

st.write('')
st.write('')

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

st.title("🤖 Multi-Agent AI Wall Street Team 🚀🌕")
st.markdown("""
This app use LangGraph to create a team of specialized AI agents running in parallel reporting to a supervisor agent.  

The team analyzes short, medium, and long-term horizons to provide a weighted recommendation to buy/hold/sell.
""")

with st.sidebar:
    st.header("Settings")
    api_key = st.secrets["open_ai_api_key"]
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    analyze_clicked = st.button("Analyze")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- 2. DATA UTILITIES ---
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    df_1d = stock.history(period="1d", interval="1m")
    df_1m = stock.history(period="1mo", interval="1d")
    df_1y = stock.history(period="1y", interval="1d")
    info = stock.info
    dividends = stock.dividends
    return df_1d, df_1m, df_1y, info, dividends

def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

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
llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def fundamental_node(state: AgentState):
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Data context: {state['data_summary']}."
    res = await llm.ainvoke(prompt)
    return {"fundamental_report": res.content}

async def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Data context: {state['data_summary']}."
    res = await llm.ainvoke(prompt)
    return {"technical_report": res.content}

async def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}."
    res = await llm.ainvoke(prompt)
    return {"ml_report": res.content}

async def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecaster for {state['ticker']}."
    res = await llm.ainvoke(prompt)
    return {"forecasting_report": res.content}

async def news_node(state: AgentState):
    prompt = f"Act as a Sentiment Analyst for {state['ticker']}."
    res = await llm.ainvoke(prompt)
    return {"news_report": res.content}

async def supervisor_node(state: AgentState):
    context = f"{state['fundamental_report']} {state['technical_report']} {state['ml_report']} {state['forecasting_report']} {state['news_report']}"
    res = await llm.ainvoke(context)
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
    if analyze_clicked:
        with st.spinner("Synthesizing multi-horizon recommendations..."):
            df_1d, df_1m, df_1y, info, dividends = get_financial_data(selected_ticker)

            df_1d = add_indicators(df_1d)
            df_1m = add_indicators(df_1m)
            df_1y = add_indicators(df_1y)

            summary = f"TICKER: {selected_ticker}"
            initial_state = {"ticker": selected_ticker, "data_summary": summary}
            final_state = asyncio.run(graph.ainvoke(initial_state))

            st.session_state['final_state'] = final_state
            st.session_state['ticker'] = selected_ticker

            st.header("Executive Multi-Horizon Recommendation")
            st.markdown(final_state['final_recommendation'])

    if 'final_state' in st.session_state:
        pdf_data = create_pdf(st.session_state['ticker'], st.session_state['final_state'])
        st.sidebar.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name=f"{st.session_state['ticker']}_Report.pdf",
            mime="application/pdf"
        )
```
