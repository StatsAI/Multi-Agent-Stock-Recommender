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
    if not text: return ""
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '*',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_pdf(ticker, final_state, charts):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 1. Add Logo
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_logo:
            logo.save(tmp_logo.name)
            pdf.image(tmp_logo.name, x=10, y=8, w=30)
    except: pass
    
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

    # Executive Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, " EXECUTIVE MULTI-HORIZON RECOMMENDATION", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    write_formatted_content(clean_text_for_pdf(final_state.get('final_recommendation', '')))
    
    # Charts in Executive Summary
    if charts:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Technical Market Context", ln=True)
        for fig in charts:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
                fig.write_image(tmp_chart.name, width=800, height=400)
                pdf.image(tmp_chart.name, w=170)
                pdf.ln(5)

    # Detailed Agent Reports
    reports = {
        "Fundamental Analysis": 'fundamental_report',
        "Technical Analysis": 'technical_report',
        "ML Analysis": 'ml_report',
        "Forecasting Analysis": 'forecasting_report',
        "News & Sentiment": 'news_report'
    }
    
    for title, key in reports.items():
        if key in final_state:
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, title.upper(), ln=True)
            pdf.set_font("Helvetica", "", 10)
            write_formatted_content(clean_text_for_pdf(final_state[key]))
            pdf.ln(8)
            
    return bytes(pdf.output())

# --- APP UI SETUP ---
logo = Image.open('picture.png')
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

with st.sidebar:
    st.image(logo)
    st.header("Settings")
    api_key = st.secrets.get("open_ai_api_key", "")
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    analyze_clicked = st.button("Analyze") 
    if api_key: os.environ["OPENAI_API_KEY"] = api_key

# --- LANGGRAPH LOGIC ---
class AgentState(TypedDict):
    ticker: str
    data_summary: str
    fundamental_report: str
    technical_report: str
    ml_report: str
    forecasting_report: str
    news_report: str
    final_recommendation: str

llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def fundamental_node(state: AgentState):
    res = await llm.ainvoke(f"Fundamental Analysis for {state['ticker']}. Confidence 0-100%.")
    return {"fundamental_report": res.content}

async def technical_node(state: AgentState):
    res = await llm.ainvoke(f"Technical Analysis for {state['ticker']}. Confidence 0-100%.")
    return {"technical_report": res.content}

async def ml_node(state: AgentState):
    res = await llm.ainvoke(f"ML Pattern Recognition for {state['ticker']}. Confidence 0-100%.")
    return {"ml_report": res.content}

async def forecasting_node(state: AgentState):
    res = await llm.ainvoke(f"Price Forecasting for {state['ticker']}. Confidence 0-100%.")
    return {"forecasting_report": res.content}

async def news_node(state: AgentState):
    res = await llm.ainvoke(f"Sentiment Analysis for {state['ticker']}. Confidence 0-100%.")
    return {"news_report": res.content}

async def supervisor_node(state: AgentState):
    # Ensure all keys are accessed safely
    reports = [state.get(k, "N/A") for k in ['fundamental_report', 'technical_report', 'ml_report', 'forecasting_report', 'news_report']]
    prompt = f"Synthesize these reports for {state['ticker']}: {' | '.join(reports)}"
    res = await llm.ainvoke(prompt)
    return {"final_recommendation": res.content}

# --- CORRECTED GRAPH FLOW ---
builder = StateGraph(AgentState)
builder.add_node("fundamental", fundamental_node)
builder.add_node("technical", technical_node)
builder.add_node("ml", ml_node)
builder.add_node("forecasting", forecasting_node)
builder.add_node("news", news_node)
builder.add_node("supervisor", supervisor_node)

builder.set_entry_point("fundamental")
# Forces parallel execution but joins at the supervisor
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

# --- EXECUTION ---
if not api_key:
    st.info("Please enter your OpenAI API Key.")
else:
    if analyze_clicked:
        with st.spinner("Analyzing..."):
            stock = yf.Ticker(selected_ticker)
            df = stock.history(period="1y")
            
            def make_chart(df):
                fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], name="Price"))
                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0))
                return fig
            
            charts = [make_chart(df)] * 3 # Placeholder for logic
            state = {"ticker": selected_ticker, "data_summary": "Data loaded."}
            final_res = asyncio.run(graph.ainvoke(state))

            st.session_state['final_state'] = final_res
            st.session_state['ticker'] = selected_ticker
            st.session_state['charts'] = charts

    if st.session_state['final_state']:
        st.markdown(st.session_state['final_state']['final_recommendation'])
        pdf_data = create_pdf(st.session_state['ticker'], st.session_state['final_state'], st.session_state['charts'])
        st.sidebar.download_button("Download Report", data=pdf_data, file_name=f"{st.session_state['ticker']}_Report.pdf")
