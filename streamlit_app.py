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
import tempfile

# --- INITIALIZE SESSION STATE ---
if 'final_state' not in st.session_state:
    st.session_state['final_state'] = None
if 'charts' not in st.session_state:
    st.session_state['charts'] = []
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = ""

# --- PDF UTILITIES ---
def clean_text_for_pdf(text):
    """Replaces common unicode characters that cause Helvetica to crash."""
    replacements = {
        '\u2013': '-', # en dash
        '\u2014': '-', # em dash
        '\u2018': "'", # left single quote
        '\u2019': "'", # right single quote
        '\u201c': '"', # left double quote
        '\u201d': '"', # right double quote
        '\u2022': '*', # bullet point
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_pdf(ticker, final_state, charts):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 1. Add Logo to top
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_logo:
            logo.save(tmp_logo.name)
            pdf.image(tmp_logo.name, x=10, y=8, w=30)
    except Exception as e:
        pass # Fallback if logo fails
    
    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, f"AI Wall Street Report: {ticker}", ln=True, align='C')
    pdf.set_draw_color(50, 50, 50)
    pdf.line(10, 30, 200, 30)
    pdf.ln(15)
    
    def write_formatted_content(text):
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            if line.startswith('###'):
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, line.replace('###', '').strip(), ln=True)
                pdf.set_font("Helvetica", "", 10)
            elif line.startswith('* ') or line.startswith('- '):
                pdf.set_x(15)
                pdf.cell(5, 5, chr(149), ln=0)
                pdf.multi_cell(0, 5, line[2:].strip())
            else:
                pdf.multi_cell(0, 5, line)

    # Section: Executive Recommendation
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, " EXECUTIVE MULTI-HORIZON RECOMMENDATION", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    write_formatted_content(clean_text_for_pdf(final_state['final_recommendation']))
    pdf.ln(5)

    # 2. Add Charts to Executive Summary
    if charts:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Technical Market Context", ln=True)
        for fig in charts:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                fig.write_image(tmp_chart.name, width=800, height=400)
                pdf.image(tmp_chart.name, w=180)
                pdf.ln(5)
    
    # Agent Sections
    reports = {
        "Fundamental Analysis": 'fundamental_report',
        "Technical Analysis": 'technical_report',
        "ML Analysis": 'ml_report',
        "Forecasting Analysis": 'forecasting_report',
        "News & Sentiment": 'news_report'
    }
    
    for title, key in reports.items():
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, title.upper(), ln=True)
        pdf.set_font("Helvetica", "", 10)
        content = clean_text_for_pdf(final_state[key])
        write_formatted_content(content)
        pdf.ln(8)
        
    return bytes(pdf.output())

# Load Logo
logo = Image.open('picture.png')

# Sidebar Styling
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

# --- SETTINGS & UI ---
st.set_page_config(page_title="AI Wall Street Team", layout="wide")
st.title("🤖 Multi-Agent AI Wall Street Team 🚀🌕")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.secrets.get("open_ai_api_key", "")
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    analyze_clicked = st.button("Analyze") 
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- DATA UTILITIES ---
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

# --- LANGGRAPH AGENT STATE ---
class AgentState(TypedDict):
    ticker: str
    data_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    forecasting_report: str
    news_report: str
    final_recommendation: str

# --- ASYNC AGENT NODES ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def fundamental_node(state: AgentState):
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Analyze long-term health. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"fundamental_report": res.content}

async def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Analyze price action using RSI/SMA. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"technical_report": res.content}

async def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}. Identify patterns. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"ml_report": res.content}

async def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecaster for {state['ticker']}. Provide price targets. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"forecasting_report": res.content}

async def news_node(state: AgentState):
    prompt = f"Act as a Sentiment Analyst for {state['ticker']}. Evaluate news impact. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"news_report": res.content}

async def supervisor_node(state: AgentState):
    context = f"Reports: {state['fundamental_report']} | {state['technical_report']} | {state['ml_report']} | {state['forecasting_report']} | {state['news_report']}"
    prompt = f"As CIO, synthesize all reports for {state['ticker']}. Provide clear recommendations and confidence scores for 1 day, 1 week, 1 month, and 6 months."
    res = await llm.ainvoke(prompt + context)
    return {"final_recommendation": res.content}

# --- GRAPH CONSTRUCTION ---
builder = StateGraph(AgentState)
builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("ml", ml_node)
builder.add_node("forecasting", forecasting_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)
builder.set_entry_point("fundamental")
for node in ["fundamental", "technical", "ml", "forecasting", "news"]:
    builder.add_edge(node, "supervisor")
builder.add_edge("supervisor", END)
graph = builder.compile()

# --- APP EXECUTION ---
if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
else:
    if analyze_clicked:
        with st.spinner("Synthesizing multi-horizon recommendations..."):
            df_1d, df_1m, df_1y, info, dividends = get_financial_data(selected_ticker)
            
            df_1d, df_1m, df_1y = add_indicators(df_1d), add_indicators(df_1m), add_indicators(df_1y)

            def create_technical_chart(df, title, is_candle=True):
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                if is_candle:
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name="SMA 20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"), row=2, col=1)
                fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                return fig

            c1, c2, c3 = create_technical_chart(df_1d, "Short"), create_technical_chart(df_1m, "Medium"), create_technical_chart(df_1y, "Long", False)
            
            summary = f"TICKER: {selected_ticker} | Close: {df_1d['Close'].iloc[-1]:.2f}"
            initial_state = {"ticker": selected_ticker, "data_summary": summary}
            final_res = asyncio.run(graph.ainvoke(initial_state))

            # Store in session state
            st.session_state['final_state'] = final_res
            st.session_state['ticker'] = selected_ticker
            st.session_state['charts'] = [c1, c2, c3]

    # Render results if they exist
    if st.session_state['final_state']:
        st.header(f"Executive Recommendation for {st.session_state['ticker']}")
        st.markdown(st.session_state['final_state']['final_recommendation'])
        
        # Display charts in app
        cols = st.columns(3)
        for idx, chart in enumerate(st.session_state['charts']):
            cols[idx].plotly_chart(chart, use_container_width=True)

        # PDF Download
        pdf_data = create_pdf(st.session_state['ticker'], st.session_state['final_state'], st.session_state['charts'])
        st.sidebar.download_button("Download PDF Report", data=pdf_data, file_name=f"{st.session_state['ticker']}_Report.pdf", mime="application/pdf")
