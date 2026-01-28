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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import re
import streamlit.components.v1 as components

# --- PDF UTILITIES ---
def create_pdf(ticker, final_state):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    elements = []
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
    styles.add(ParagraphStyle(name='BulletPoint',
                             parent=styles['BodyText'],
                             leftIndent=20,
                             bulletIndent=10,
                             spaceAfter=6))
    
    try:
        logo = RLImage('picture.png', width=2*inch, height=2*inch)
        logo.hAlign = 'CENTER'
        elements.append(logo)
        elements.append(Spacer(1, 12))
    except:
        pass
    
    title = Paragraph(f"AI Wall Street Report: {ticker}", styles['CustomTitle'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    def format_bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    def format_content(content):
        lines = content.split('\n')
        formatted_elements = []
        for line in lines:
            line = line.strip()
            if not line:
                formatted_elements.append(Spacer(1, 6))
                continue
            line = format_bold(line)
            if line.startswith('###'):
                header_text = line.replace('###', '').strip()
                formatted_elements.append(Paragraph(f"<b>{header_text}</b>", styles['BodyText']))
            elif line.startswith('* ') or line.startswith('- '):
                bullet_text = line[2:].strip()
                formatted_elements.append(Paragraph(f"• {bullet_text}", styles['BulletPoint']))
            else:
                formatted_elements.append(Paragraph(line, styles['BodyText']))
        return formatted_elements
    
    exec_header = Paragraph("EXECUTIVE MULTI-HORIZON RECOMMENDATION", styles['SectionHeader'])
    elements.append(exec_header)
    elements.extend(format_content(final_state['final_recommendation']))
    elements.append(Spacer(1, 20))
    
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
        elements.extend(format_content(final_state[key]))
        elements.append(Spacer(1, 20))
    
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# Set page config
st.set_page_config(page_title="AI Wall Street Team", layout="wide")

# --- CUSTOM CSS FOR BUTTON UNIFORMITY ---
st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:has(button) {
        align-items: stretch;
    }
    .stButton > button {
        height: 3em; 
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

try:
    logo_img = Image.open('picture.png')
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
        st.image(logo_img)
except:
    pass

st.markdown("""
        <style>
                .block-container {
                padding-top: 0;
                }
        </style>
        """, unsafe_allow_html=True)

st.title("🤖 Multi-Agent AI Wall Street Team 🚀🌕")
st.markdown("""
This app uses LangGraph to create a team of specialized AI agents running in parallel reporting to a supervisor agent.  
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.secrets.get("open_ai_api_key", "")
    selected_ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_clicked = st.button("Analyze", use_container_width=True) 
    with col2:
        if st.button("Clear Results", use_container_width=True):
            for key in ['final_state', 'ticker', 'df_1d', 'df_1m', 'df_1y']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
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
    if df.empty: return df
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
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

async def fundamental_node(state: AgentState):
    prompt = f"Act as a Fundamental Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze long-term health. Position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"fundamental_report": res.content}

async def technical_node(state: AgentState):
    prompt = f"Act as a Technical Analyst for {state['ticker']}. Data context: {state['data_summary']}. Analyze RSI/SMA. Position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"technical_report": res.content}

async def ml_node(state: AgentState):
    prompt = f"Act as an ML Analyst for {state['ticker']}. Identify patterns. Position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"ml_report": res.content}

async def forecasting_node(state: AgentState):
    prompt = f"Act as a Time Series Forecaster for {state['ticker']}. Position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"forecasting_report": res.content}

async def news_node(state: AgentState):
    prompt = f"Act as a Sentiment Analyst for {state['ticker']}. Evaluate macro impact. Position with Confidence Score (0-100%)."
    res = await llm.ainvoke(prompt)
    return {"news_report": res.content}

async def supervisor_node(state: AgentState):
    context = f"Reports: {state['fundamental_report']} | {state['technical_report']} | {state['ml_report']} | {state['forecasting_report']} | {state['news_report']}"
    prompt = f"As CIO, synthesize all reports for {state['ticker']}. Clear recommendation (Buy/Hold/Sell) for 4 horizons: Day, Week, Month, 6 Months."
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

# --- 6. APP EXECUTION ---
if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
else:
    if analyze_clicked:
        with st.spinner("Fetching data and generating charts..."):
            df_1d, df_1m, df_1y, info, dividends = get_financial_data(selected_ticker)
            df_1d, df_1m, df_1y = add_indicators(df_1d), add_indicators(df_1m), add_indicators(df_1y)
            
            st.caption("1-Day Intraday with SMA & RSI (Short)")
            st.plotly_chart(create_technical_chart(df_1d, "Short"), use_container_width=True)
            st.caption("1-Month Daily with SMA & RSI (Medium)")
            st.plotly_chart(create_technical_chart(df_1m, "Medium"), use_container_width=True)
            st.caption("1-Year Daily with SMA & RSI (Long)")
            st.plotly_chart(create_technical_chart(df_1y, "Long", is_candle=False), use_container_width=True)

        with st.spinner("Synthesizing multi-horizon recommendations..."):
            summary = (f"TICKER: {selected_ticker}\n"
                       f"CURRENT CLOSE: {df_1d['Close'].iloc[-1]:.2f}")

            initial_state = {"ticker": selected_ticker, "data_summary": summary}
            final_state = asyncio.run(graph.ainvoke(initial_state))

            st.session_state['final_state'] = final_state
            st.session_state['ticker'] = selected_ticker
            st.session_state['df_1d'] = df_1d
            st.session_state['df_1m'] = df_1m
            st.session_state['df_1y'] = df_1y
            st.rerun()

    if 'final_state' in st.session_state and 'df_1d' in st.session_state:
        st.caption("1-Day Intraday with SMA & RSI (Short)")
        st.plotly_chart(create_technical_chart(st.session_state['df_1d'], "Short"), use_container_width=True)
        st.caption("1-Month Daily with SMA & RSI (Medium)")
        st.plotly_chart(create_technical_chart(st.session_state['df_1m'], "Medium"), use_container_width=True)
        st.caption("1-Year Daily with SMA & RSI (Long)")
        st.plotly_chart(create_technical_chart(st.session_state['df_1y'], "Long", is_candle=False), use_container_width=True)

        st.header("Executive Multi-Horizon Recommendation")
        st.markdown(st.session_state['final_state']['final_recommendation'])
        
        st.divider()
        tabs = st.tabs(["Fundamental", "Technical", "ML", "Forecasting", "News"])
        reports = ['fundamental_report', 'technical_report', 'ml_report', 'forecasting_report', 'news_report']
        for i, tab in enumerate(tabs):
            with tab: st.write(st.session_state['final_state'][reports[i]])

    if 'final_state' in st.session_state:
        pdf_data = create_pdf(st.session_state['ticker'], st.session_state['final_state'])
        st.sidebar.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name=f"{st.session_state['ticker']}_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        # --- EXPLAINER SECTION ---
        if st.sidebar.button("Explain how the app works!", use_container_width=True):
            st.divider()
            st.header("App Architecture & Agentic Workflow")
            
            # 1. Logical Architecture (Diagram) First
            st.subheader("1. Logical Architecture")
            mermaid_code = """
            graph TD
                U[User Input] -->|Ticker| DI[Data Ingestion Node]
                DI -->|Price/Info| S{Agent State}
                
                subgraph Analysts [Parallel Analysis Layer]
                    S --> FA[Fundamental Agent]
                    S --> TA[Technical Agent]
                    S --> ML[ML Agent]
                    S --> TSA[Forecasting Agent]
                    S --> SA[News/Sentiment Agent]
                end
                
                FA -->|Report| SUP[Supervisor: CIO Agent]
                TA -->|Report| SUP
                ML -->|Report| SUP
                TSA -->|Report| SUP
                SA -->|Report| SUP
                
                SUP -->|Final Synthesis| OUT[Multi-Horizon Strategy]
                
                style S fill:#f9f,stroke:#333,stroke-width:2px
                style Analysts fill:#f0f0f0,stroke:#666,stroke-dasharray: 5 5
                style SUP fill:#4b0082,color:#fff,stroke-width:2px
                style OUT fill:#00c853,color:#fff
            """
            
            html_code = f"""
            <div class="mermaid" style="background-color: white; padding: 20px; border-radius: 15px; border: 1px solid #ddd;">
                {mermaid_code}
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'forest' }});
            </script>
            """
            components.html(html_code, height=550)

            # 2. Detailed Process Overview
            st.subheader("2. Process Overview")
            st.markdown("""
            1. **Data Ingestion & Normalization**: The system triggers `yfinance` to fetch real-time intraday (1m), monthly (1d), and yearly (1d) price data. It also extracts key-value pairs from the company's financial info (PE ratio, Market Cap, etc.) to create a structured **Data Summary**.
            
            2. **The LangGraph State Machine**: A specialized `AgentState` object is initialized. This acts as the "shared memory" for the entire system, ensuring that every agent has access to the same raw data while contributing their specific analysis back to the pool.
            
            3. **Parallel Multi-Agent Execution**:
                * **Fundamental Agent**: Deep-dives into long-term valuation and company health.
                * **Technical Agent**: Calculates RSI, SMA crossovers, and momentum indicators.
                * **ML Agent**: Uses pattern recognition to identify historical price-action clusters.
                * **Forecasting Agent**: Acts as a time-series expert to project potential price ranges.
                * **News Agent**: Evaluates macro-sentiment and recent headline impact.
            
            4. **Conflict Resolution & Synthesis**: Each agent produces a report with a **Confidence Score**. The **Supervisor (CIO) Agent** reviews these five reports. If the Technical Agent is "Bullish" but the Fundamental Agent is "Bearish," the Supervisor weighs the evidence based on the investment horizon to provide a balanced final verdict.
            
            5. **Final Output Generation**: The final strategy is rendered in the UI and converted into a professional PDF report using `ReportLab`, complete with executive summaries and detailed analyst breakdowns.
            """)
