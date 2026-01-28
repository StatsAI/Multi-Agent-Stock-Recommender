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
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO

# --- PDF UTILITIES ---
def create_pdf(ticker, final_state):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', 
                             parent=styles['Heading1'],
                             fontSize=24,
                             textColor='black',
                             spaceAfter=30,
                             alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='SectionHeader',
                             parent=styles['Heading2'],
                             fontSize=14,
                             textColor='black',
                             spaceAfter=12,
                             spaceBefore=12,
                             backColor='#f0f0f0'))
    
    # Title
    title = Paragraph(f"AI Wall Street Report: {ticker}", styles['CustomTitle'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Executive Recommendation Section
    exec_header = Paragraph("EXECUTIVE MULTI-HORIZON RECOMMENDATION", styles['SectionHeader'])
    elements.append(exec_header)
    
    exec_text = final_state['final_recommendation'].replace('\n', '<br/>')
    exec_para = Paragraph(exec_text, styles['BodyText'])
    elements.append(exec_para)
    elements.append(Spacer(1, 20))
    
    # Agent Sections
    reports = {
        "FUNDAMENTAL ANALYSIS": 'fundamental_report',
        "TECHNICAL ANALYSIS": 'technical_report',
        "ML ANALYSIS": 'ml_report',
        "FORECASTING ANALYSIS": 'forecasting_report',
        "NEWS & SENTIMENT": 'news_report'
    }
    
    for title, key in reports.items():
        section_header = Paragraph(title, styles['SectionHeader'])
        elements.append(section_header)
        
        content = final_state[key].replace('\n', '<br/>')
        section_para = Paragraph(content, styles['BodyText'])
        elements.append(section_para)
        elements.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

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

# Sidebar for Configuration
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
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze long-term health. Provide insights for the next month and 6 months specifically. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"fundamental_report": res.content}

async def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze price action for the next day and next week using RSI/SMA. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"technical_report": res.content}

async def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}. Identify patterns for the next day, week, and month. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"ml_report": res.content}

async def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecaster for {state['ticker']}. Provide price targets for next day, week, month, and 6 months. State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"forecasting_report": res.content}

async def news_node(state: AgentState):
    prompt = f"Act as a Sentiment Analyst for {state['ticker']}. Evaluate macro impact on all horizons (Day to 6-Months). State a position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"news_report": res.content}

async def supervisor_node(state: AgentState):
    context = f"Reports: {state['fundamental_report']} | {state['technical_report']} | {state['ml_report']} | {state['forecasting_report']} | {state['news_report']}"
    prompt = (
        f"As CIO, synthesize all reports for {state['ticker']}. You must provide a clear recommendation (Buy/Hold/Sell) "
        f"for four specific horizons: 1. Next Day, 2. Next Week, 3. Next Month, 4. Next 6 Months. "
        f"For EACH horizon, provide: \n"
        f"- The Recommendation\n"
        f"- A Confidence Score (0-100%)\n"
        f"- A detailed explanation of how that score was arrived at based on the agent reports."
    )
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
    if analyze_clicked:
        with st.spinner("Synthesizing multi-horizon recommendations..."):
            df_1d, df_1m, df_1y, info, dividends = get_financial_data(selected_ticker)
            
            df_1d = add_indicators(df_1d)
            df_1m = add_indicators(df_1m)
            df_1y = add_indicators(df_1y)

            def create_technical_chart(df, title, is_candle=True):
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                if is_candle:
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='cyan'), name="Price"), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name="SMA 20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0,r=0,b=0,t=0), showlegend=False)
                return fig

            st.caption("1-Day Intraday with SMA & RSI (Short)")
            st.plotly_chart(create_technical_chart(df_1d, "Short"), use_container_width=True)
            st.caption("1-Month Daily with SMA & RSI (Medium)")
            st.plotly_chart(create_technical_chart(df_1m, "Medium"), use_container_width=True)
            st.caption("1-Year Daily with SMA & RSI (Long)")
            st.plotly_chart(create_technical_chart(df_1y, "Long", is_candle=False), use_container_width=True)

            div_summary = dividends.tail(5).to_string() if not dividends.empty else "No dividends"
            summary = (f"TICKER: {selected_ticker}\n"
                       f"INFO: P/E: {info.get('trailingPE')}, Mkt Cap: {info.get('marketCap')}, Div Yield: {info.get('dividendYield')}\n"
                       f"DIVIDENDS (Recent): {div_summary}\n"
                       f"CURRENT CLOSE: {df_1d['Close'].iloc[-1]:.2f}\n"
                       f"1M RANGE: {df_1m['Low'].min():.2f} - {df_1m['High'].max():.2f}\n"
                       f"1Y TREND: Start {df_1y['Close'].iloc[0]:.2f} to End {df_1y['Close'].iloc[-1]:.2f}")

            initial_state = {"ticker": selected_ticker, "data_summary": summary}
            final_state = asyncio.run(graph.ainvoke(initial_state))

            # Store final state in session for PDF generation button
            st.session_state['final_state'] = final_state
            st.session_state['ticker'] = selected_ticker

            st.header("Executive Multi-Horizon Recommendation")
            st.markdown(final_state['final_recommendation'])
            
            st.divider()
            tabs = st.tabs(["Fundamental", "Technical", "ML", "Forecasting", "News"])
            reports = ['fundamental_report', 'technical_report', 'ml_report', 'forecasting_report', 'news_report']
            for i, tab in enumerate(tabs):
                with tab: st.write(final_state[reports[i]])

    # Sidebar PDF Download Button
    if 'final_state' in st.session_state:
        pdf_data = create_pdf(st.session_state['ticker'], st.session_state['final_state'])
        st.sidebar.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name=f"{st.session_state['ticker']}_Report.pdf",
            mime="application/pdf"
        )
