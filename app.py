import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from generate_posts import generate_posts
from nlp_engine import analyze_posts
from network_analysis import (
    build_interaction_network, detect_communities,
    compute_user_risk, get_network_layout,
)

st.set_page_config(
    page_title="SOCMINT Analyzer",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN TOKENS (White / Black / Red / Green) ─────────────────────────────
BG       = "#ffffff"
BG2      = "#f8f8f8"
SURFACE  = "#ffffff"
SURFACE2 = "#f2f2f2"
BORDER   = "#e5e5e5"
BORDER2  = "#d4d4d4"
NAVY     = "#111111"
NAVY2    = "#111111"
TEXT1    = "#111111"
TEXT2    = "#444444"
TEXT3    = "#888888"
TEXT4    = "#bbbbbb"
ACCENT   = "#111111"
ACCENT2  = "#dc2626"
BLUE     = "#888888"
GREEN    = "#16a34a"
AMBER    = "#dc2626"
RED      = "#dc2626"
ORANGE   = "#dc2626"
VIOLET   = "#444444"
CYAN     = "#888888"

THREAT_COLORS = {"High": "#dc2626", "Medium": "#f87171", "Low": "#888888", "None": "#e5e5e5"}
RISK_SCALE = [[0, "#e5e5e5"], [0.3, "#888888"], [0.6, "#f87171"], [1, "#dc2626"]]
SERIES = ["#dc2626", "#111111", "#888888", "#16a34a", "#f87171", "#444444", "#bbbbbb"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Lato, sans-serif", color="#444444", size=12),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(gridcolor="#e5e5e5", tickfont=dict(family="Lato, sans-serif", color="#888888", size=11), linecolor="#e5e5e5"),
    yaxis=dict(gridcolor="#e5e5e5", tickfont=dict(family="Lato, sans-serif", color="#888888", size=11), linecolor="#e5e5e5"),
    legend=dict(font=dict(color="#444444", size=11), bgcolor="rgba(0,0,0,0)"),
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Lato:wght@300;400;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Lato', sans-serif;
    background-color: {BG2};
    color: {TEXT2};
  }}
  .main {{ background: {BG2}; }}
  h1, h2, h3 {{
    font-family: 'Montserrat', sans-serif !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
  }}

  /* ── Top Bar ── */
  .top-bar {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
    padding: 14px 0 14px 0;
    margin: -1rem -1rem 24px -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .top-bar-left {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .top-bar-logo {{
    width: 36px;
    height: 36px;
    background: #111111;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 16px;
    font-family: 'Montserrat', sans-serif;
  }}
  .top-bar-title {{
    font-family: 'Montserrat', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: {NAVY};
  }}
  .top-bar-subtitle {{
    font-family: 'Lato', sans-serif;
    font-size: 12px;
    color: {TEXT3};
    font-weight: 400;
  }}
  .top-bar-right {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}

  /* ── Status Badge ── */
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
  }}
  .status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Lato', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #16a34a;
    background: rgba(22, 163, 74, 0.06);
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(22, 163, 74, 0.15);
  }}
  .status-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #16a34a;
    animation: pulse 2s ease-in-out infinite;
  }}

  /* ── Section Headers ── */
  .section-header {{
    font-family: 'Montserrat', sans-serif;
    color: {NAVY};
    font-size: 14px;
    font-weight: 600;
    padding-bottom: 10px;
    margin-bottom: 16px;
    border-bottom: 2px solid #111111;
    display: inline-block;
  }}
  .section-sub {{
    font-family: 'Lato', sans-serif;
    font-size: 12px;
    color: {TEXT3};
    margin-bottom: 16px;
  }}

  /* ── Metric Cards ── */
  .metric-card {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s ease;
  }}
  .metric-card:hover {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }}
  .metric-card.accent-left {{
    border-left: 4px solid #111111;
  }}
  .metric-card.accent-red {{
    border-left: 4px solid #dc2626;
  }}
  .metric-card.accent-amber {{
    border-left: 4px solid #f87171;
  }}
  .metric-card.accent-blue {{
    border-left: 4px solid #16a34a;
  }}

  .metric-label {{
    font-family: 'Lato', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {TEXT3};
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-family: 'Montserrat', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
  }}
  .metric-delta {{
    font-family: 'Lato', sans-serif;
    font-size: 12px;
    font-weight: 400;
    margin-top: 8px;
    color: {TEXT3};
  }}
  .metric-delta.danger  {{ color: #dc2626; }}
  .metric-delta.warning {{ color: #dc2626; }}
  .metric-delta.success {{ color: #16a34a; }}
  .metric-delta.info    {{ color: #16a34a; }}
  .metric-delta.accent  {{ color: #111111; }}

  /* ── Risk Badges ── */
  .risk-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Lato', sans-serif;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  .risk-badge.critical {{ background: rgba(220,38,38,0.08); color: #dc2626; }}
  .risk-badge.high     {{ background: rgba(220,38,38,0.08); color: #dc2626; }}
  .risk-badge.medium   {{ background: rgba(220,38,38,0.05); color: #f87171; }}
  .risk-badge.low      {{ background: rgba(0,0,0,0.04);  color: #888888; }}

  /* ── Alert Cards ── */
  .alert-feed {{
    max-height: 700px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: {BORDER2} transparent;
    padding-right: 4px;
  }}
  .alert-card {{
    background: {BG};
    border: 1px solid {BORDER};
    border-left: 4px solid {TEXT4};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 13px;
    line-height: 1.6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .alert-card.severity-high   {{ border-left-color: #dc2626; }}
  .alert-card.severity-medium {{ border-left-color: #f87171; }}
  .alert-card.severity-low    {{ border-left-color: #bbbbbb; }}

  .alert-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }}
  .alert-user {{
    color: {NAVY};
    font-weight: 700;
    font-size: 13px;
  }}
  .alert-meta {{
    color: {TEXT3};
    font-size: 11px;
  }}
  .alert-score {{
    margin-left: auto;
    font-family: 'Montserrat', sans-serif;
    font-size: 14px;
    font-weight: 700;
  }}
  .alert-text {{
    color: {TEXT2};
    font-size: 14px;
    margin-bottom: 12px;
    line-height: 1.6;
    font-style: italic;
  }}
  .alert-tags {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    color: {TEXT3};
    font-size: 11px;
  }}

  /* ── Community Cards ── */
  .community-card {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}

  /* ── Stat Block ── */
  .stat-block {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid {BORDER};
  }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-label {{ color: {TEXT3}; font-size: 12px; font-weight: 400; }}
  .stat-value {{ color: {NAVY}; font-family: 'Montserrat', sans-serif; font-size: 14px; font-weight: 600; }}

  /* ── Intel Cards (Recorded Future style) ── */
  .intel-card {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .intel-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid {BORDER};
  }}
  .intel-card-name {{
    font-family: 'Montserrat', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: {NAVY};
  }}
  .risk-score-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 12px;
    font-family: 'Montserrat', sans-serif;
    font-size: 20px;
    font-weight: 700;
  }}
  .risk-score-badge.critical {{ background: rgba(220,38,38,0.08); color: #dc2626; }}
  .risk-score-badge.high     {{ background: rgba(220,38,38,0.08); color: #dc2626; }}
  .risk-score-badge.medium   {{ background: rgba(220,38,38,0.05); color: #f87171; }}
  .risk-score-badge.low      {{ background: rgba(0,0,0,0.04);  color: #888888; }}

  .intel-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid {BG2};
  }}
  .intel-row:last-child {{ border-bottom: none; }}
  .intel-label {{ color: {TEXT3}; }}
  .intel-value {{ color: {NAVY}; font-weight: 600; }}

  /* ── Keyword Bars ── */
  .kw-bar-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 5px 0;
    font-size: 12px;
  }}
  .kw-bar-label {{ color: {TEXT2}; min-width: 90px; font-weight: 400; }}
  .kw-bar-track {{ flex: 1; background: {SURFACE2}; height: 6px; border-radius: 3px; }}
  .kw-bar-fill {{ height: 6px; border-radius: 3px; background: #dc2626; }}
  .kw-bar-count {{ color: {TEXT3}; min-width: 28px; text-align: right; }}

  /* ── Sidebar ── */
  .stSidebar {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
  }}

  /* ── Buttons ── */
  .stButton button {{
    background: #111111;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 12px;
    padding: 10px 20px;
    letter-spacing: 0.02em;
  }}
  .stButton button:hover {{ background: #333333; }}

  /* ── Form Labels ── */
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stSelectbox"] label,
  div[data-testid="stRadio"] label {{
    color: {NAVY};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'Lato', sans-serif;
  }}

  .stDataFrame {{ font-family: 'Lato', sans-serif; font-size: 12px; }}

  hr {{ border-color: {BORDER} !important; opacity: 0.6; }}
  .stRadio > div {{ gap: 2px; }}

  .stDownloadButton button {{
    background: transparent;
    color: #111111;
    border: 1px solid #d4d4d4;
    border-radius: 8px;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 12px;
  }}
  .stDownloadButton button:hover {{ background: #111111; color: white; }}

  /* ── Chart Containers ── */
  .chart-container {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 16px;
  }}

  /* ── Timestamp ── */
  .ts {{ font-size: 11px; color: {TEXT3}; }}
</style>
""", unsafe_allow_html=True)

# ── TOP BAR ─────────────────────────────────��────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <div class="top-bar-logo">S</div>
        <div>
            <div class="top-bar-title">SOCMINT Analyzer</div>
            <div class="top-bar-subtitle">Social Media Intelligence Platform</div>
        </div>
    </div>
    <div class="top-bar-right">
        <span class="ts">Last scan: {now_str}</span>
        <div class="status-badge">
            <div class="status-dot"></div>
            LIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────��───────
with st.sidebar:
    st.markdown(f'<p style="font-family:Montserrat,sans-serif; font-size:11px; font-weight:700; color:{NAVY}; letter-spacing:0.1em; text-transform:uppercase; border-bottom:2px solid #111111; padding-bottom:8px; display:inline-block;">Controls</p>', unsafe_allow_html=True)

    n_posts = st.slider("POST VOLUME", 500, 5000, 2000, step=250)

    if st.button("Ingest & Analyze", use_container_width=True):
        st.session_state["run_seed"] = int(time.time())
        st.cache_data.clear()

    st.markdown("---")
    st.markdown(f'<p style="font-family:Montserrat,sans-serif; font-size:11px; font-weight:700; color:{NAVY}; letter-spacing:0.1em; text-transform:uppercase; border-bottom:2px solid #111111; padding-bottom:8px; display:inline-block;">Filters</p>', unsafe_allow_html=True)

    threat_filter = st.multiselect(
        "THREAT LEVEL",
        ["High", "Medium", "Low", "None"],
        default=["High", "Medium", "Low", "None"],
    )

    st.markdown("---")
    st.markdown(f'<p style="font-family:Montserrat,sans-serif; font-size:11px; font-weight:700; color:{NAVY}; letter-spacing:0.1em; text-transform:uppercase; border-bottom:2px solid #111111; padding-bottom:8px; display:inline-block;">View</p>', unsafe_allow_html=True)

    view_mode = st.radio(
        "DASHBOARD VIEW",
        ["Overview", "Alert Feed", "Network", "User Intel"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 11px; color: {TEXT3}; line-height: 1.8;">
        <div>Status: <span style="color:#16a34a; font-weight:700;">Operational</span></div>
        <div>Engine: NLP + NetworkX</div>
        <div>Version: 1.0.0</div>
        <div style="margin-top:8px; color:{TEXT4};">Built by Kimora Taylor</div>
    </div>
    """, unsafe_allow_html=True)

# ── DATA PIPELINE ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(n, seed):
    np.random.seed(seed)
    import random; random.seed(seed)
    df = generate_posts(n)
    df = analyze_posts(df)
    return df

@st.cache_data(show_spinner=False)
def run_network(n, seed):
    df = run_pipeline(n, seed)
    G = build_interaction_network(df)
    community_map, communities = detect_communities(G)
    user_risk = compute_user_risk(G, df, community_map)
    pos = get_network_layout(G)
    return G, community_map, communities, user_risk, pos

current_seed = st.session_state.get("run_seed", 42)

with st.spinner("Analyzing posts..."):
    df = run_pipeline(n_posts, current_seed)

filtered = df[df["threat_level"].isin(threat_filter)]
flagged = filtered[filtered["threat_level"] != "None"]

# ── HELPERS ──────────────────────────────────────────────────────────────────
def risk_score_color(score):
    if score >= 0.6: return RED
    if score >= 0.3: return AMBER
    return BLUE

# ── KPI STRIP ────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

total = len(filtered)
flagged_count = len(flagged)
high_count = len(filtered[filtered["threat_level"] == "High"])
med_count = len(filtered[filtered["threat_level"] == "Medium"])
unique_flagged_users = flagged["user"].nunique()
pct = (flagged_count / total * 100) if total else 0

with m1:
    st.markdown(f"""<div class="metric-card accent-left">
        <div class="metric-label">Posts Analyzed</div>
        <div class="metric-value">{total:,}</div>
        <div class="metric-delta accent">{n_posts:,} ingested</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card accent-amber">
        <div class="metric-label">Flagged</div>
        <div class="metric-value">{flagged_count:,}</div>
        <div class="metric-delta warning">{pct:.1f}% detection rate</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card accent-red">
        <div class="metric-label">High Threat</div>
        <div class="metric-value">{high_count}</div>
        <div class="metric-delta danger">{"requires review" if high_count > 0 else "all clear"}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card accent-amber">
        <div class="metric-label">Medium Threat</div>
        <div class="metric-value">{med_count}</div>
        <div class="metric-delta warning">monitoring</div>
    </div>""", unsafe_allow_html=True)
with m5:
    st.markdown(f"""<div class="metric-card accent-blue">
        <div class="metric-label">Flagged Accounts</div>
        <div class="metric-value">{unique_flagged_users}</div>
        <div class="metric-delta info">unique users</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Overview":
    # Full-width timeline
    st.markdown('<p class="section-header">Threat Activity Timeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">7-day activity breakdown by threat level</p>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    tl = filtered.copy()
    tl["day_hour"] = tl["timestamp"].dt.floor("6h")
    timeline = tl.groupby(["day_hour", "threat_level"]).size().reset_index(name="count")
    fig_tl = px.area(
        timeline, x="day_hour", y="count", color="threat_level",
        color_discrete_map=THREAT_COLORS, template="plotly_white",
        labels={"day_hour": "", "count": "Posts", "threat_level": ""},
    )
    fig_tl.update_layout(**PLOTLY_LAYOUT, height=220)
    fig_tl.update_traces(line=dict(width=2))
    st.plotly_chart(fig_tl, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3-column secondary
    ov1, ov2, ov3 = st.columns(3)

    with ov1:
        st.markdown('<p class="section-header">Threat Breakdown</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        tc = flagged["threat_level"].value_counts()
        if len(tc) > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=tc.index, values=tc.values, hole=0.65,
                marker_colors=[THREAT_COLORS.get(l, TEXT4) for l in tc.index],
                textfont=dict(family="Lato, sans-serif", size=11, color=TEXT2),
                hovertemplate="<b>%{label}</b><br>%{value} posts<br>%{percent}<extra></extra>",
            )])
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=8, b=0), height=260,
                legend=dict(font=dict(color=TEXT2, size=11), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ov2:
        st.markdown('<p class="section-header">Sentiment Analysis</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Histogram(
            x=filtered[filtered["threat_level"] == "None"]["sentiment_polarity"],
            name="Benign", marker_color="#d4d4d4", opacity=0.5, nbinsx=30,
        ))
        fig_sent.add_trace(go.Histogram(
            x=flagged["sentiment_polarity"],
            name="Flagged", marker_color="#dc2626", opacity=0.8, nbinsx=30,
        ))
        fig_sent.add_vline(x=0, line_dash="dot", line_color=TEXT4, opacity=0.5)
        fig_sent.update_layout(**PLOTLY_LAYOUT, height=260, barmode="overlay",
            xaxis_title="Negative ← → Positive", yaxis_title="")
        st.plotly_chart(fig_sent, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ov3:
        st.markdown('<p class="section-header">Top Flagged Accounts</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        top_users = flagged["user"].value_counts().head(8).reset_index()
        top_users.columns = ["user", "count"]
        if len(top_users) > 0:
            fig_tu = px.bar(
                top_users, x="count", y="user", orientation="h",
                color="count", color_continuous_scale=[[0, "#f2f2f2"], [1, "#dc2626"]],
                template="plotly_white", labels={"count": "", "user": ""},
            )
            fig_tu.update_layout(**PLOTLY_LAYOUT, height=260, coloraxis_showscale=False)
            fig_tu.update_layout(yaxis=dict(tickfont=dict(family="Lato, sans-serif", color=TEXT2, size=11)))
            st.plotly_chart(fig_tu, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bottom: hashtags + recent alerts
    bt1, bt2 = st.columns([1, 1.5])

    with bt1:
        st.markdown('<p class="section-header">Trending Signals</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        all_tags = " ".join(flagged["hashtags"].dropna().astype(str)).split()
        all_tags = [t for t in all_tags if t.startswith("#")]
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(10).reset_index()
            tag_counts.columns = ["hashtag", "count"]
            fig_tags = px.bar(
                tag_counts, x="count", y="hashtag", orientation="h",
                color="count", color_continuous_scale=[[0, "#f2f2f2"], [1, "#111111"]],
                template="plotly_white", labels={"count": "", "hashtag": ""},
            )
            fig_tags.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
            fig_tags.update_layout(yaxis=dict(tickfont=dict(family="Lato, sans-serif", color=TEXT2, size=11)))
            st.plotly_chart(fig_tags, use_container_width=True)
        else:
            st.info("No hashtags found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with bt2:
        st.markdown('<p class="section-header">Recent High-Threat Alerts</p>', unsafe_allow_html=True)
        recent_high = flagged[flagged["threat_level"] == "High"].sort_values("timestamp", ascending=False).head(5)
        if len(recent_high) > 0:
            for _, row in recent_high.iterrows():
                sc = risk_score_color(row["threat_score"])
                st.markdown(f"""
                <div class="alert-card severity-high">
                    <div class="alert-header">
                        <span class="risk-badge critical">HIGH</span>
                        <span class="alert-user">@{row['user']}</span>
                        <span class="alert-meta">{row['timestamp'].strftime('%m/%d %H:%M')}</span>
                        <span class="alert-score" style="color:{sc};">{row['threat_score']:.2f}</span>
                    </div>
                    <div class="alert-text">"{row['text']}"</div>
                    <div class="alert-tags">
                        <span>sentiment {row['sentiment_polarity']:.2f}</span>
                        <span>coord: {row['coordination_patterns']}</span>
                        <span>{row['likes']} likes · {row['reposts']} reposts</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No high-threat alerts.")


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT FEED
# ═══════════════════════════════════════��══════════════════════════════════════
elif view_mode == "Alert Feed":
    st.markdown('<p class="section-header">Alert Feed</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Real-time threat stream sorted by severity and score</p>', unsafe_allow_html=True)

    feed_col, detail_col = st.columns([2, 1])

    with feed_col:
        sorted_flagged = flagged.sort_values(["threat_level", "threat_score", "timestamp"],
            ascending=[True, False, False],
            key=lambda x: x.map({"High": 0, "Medium": 1, "Low": 2}) if x.name == "threat_level" else x)

        st.markdown('<div class="alert-feed">', unsafe_allow_html=True)
        for _, row in sorted_flagged.head(30).iterrows():
            sev_cls = f"severity-{row['threat_level'].lower()}"
            badge_cls = "critical" if row["threat_level"] == "High" else row["threat_level"].lower()
            sc = risk_score_color(row["threat_score"])
            kw = row["high_keywords"]
            if row["med_keywords"]:
                kw += (", " if kw else "") + row["med_keywords"]
            st.markdown(f"""
            <div class="alert-card {sev_cls}">
                <div class="alert-header">
                    <span class="risk-badge {badge_cls}">{row['threat_level']}</span>
                    <span class="alert-user">@{row['user']}</span>
                    <span class="alert-meta">{row['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
                    <span class="alert-score" style="color:{sc};">{row['threat_score']:.2f}</span>
                </div>
                <div class="alert-text">"{row['text']}"</div>
                <div class="alert-tags">
                    <span>keywords: {kw or '—'}</span>
                    <span>sentiment: {row['sentiment_polarity']:.2f}</span>
                    <span>coord: {row['coordination_patterns']}</span>
                    <span>{row['likes']} likes · {row['reposts']} reposts</span>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with detail_col:
        st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        sev_order = ["High", "Medium", "Low"]
        sev_vals = [len(flagged[flagged["threat_level"] == s]) for s in sev_order]
        fig_sev = go.Figure(go.Bar(
            x=sev_vals, y=sev_order, orientation="h",
            marker_color=[THREAT_COLORS[s] for s in sev_order],
            text=sev_vals, textposition="outside",
            textfont=dict(family="Lato, sans-serif", color=TEXT2, size=11),
        ))
        fig_sev.update_layout(**PLOTLY_LAYOUT, height=140)
        fig_sev.update_layout(yaxis=dict(tickfont=dict(family="Lato, sans-serif", color=TEXT2, size=11)))
        st.plotly_chart(fig_sev, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Detected Keywords</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        all_kw = []
        for col in ["high_keywords", "med_keywords"]:
            for val in flagged[col].dropna():
                if val:
                    all_kw.extend([k.strip() for k in val.split(",") if k.strip()])
        if all_kw:
            kw_counts = pd.Series(all_kw).value_counts().head(10)
            for kw, cnt in kw_counts.items():
                bar_w = min(cnt / kw_counts.max() * 100, 100)
                st.markdown(f"""
                <div class="kw-bar-row">
                    <span class="kw-bar-label">{kw}</span>
                    <div class="kw-bar-track">
                        <div class="kw-bar-fill" style="width:{bar_w}%;"></div>
                    </div>
                    <span class="kw-bar-count">{cnt}</span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        csv = flagged[["timestamp", "user", "text", "threat_level", "threat_score",
            "sentiment_polarity", "high_keywords", "coordination_patterns"]].to_csv(index=False)
        st.download_button(label="Export alerts (.csv)", data=csv,
            file_name="socmint_alerts.csv", mime="text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK
# ══════════════════════════════════════════════════��═══════════════════════════
elif view_mode == "Network":
    st.markdown('<p class="section-header">Interaction Network</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Conversation clusters and community detection</p>', unsafe_allow_html=True)

    with st.spinner("Building network..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(G.nodes()) == 0:
        st.warning("No network data available.")
    else:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        edge_x, edge_y = [], []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color=BORDER2), hoverinfo="none",
        )

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            comm = community_map.get(node, -1)
            risk_row = user_risk[user_risk["user"] == node]
            risk = risk_row["risk_score"].values[0] if len(risk_row) > 0 else 0
            risk_lvl = risk_row["risk_level"].values[0] if len(risk_row) > 0 else "None"
            degree = G.degree(node, weight="weight")
            node_text.append(f"<b>@{node}</b><br>Risk: {risk_lvl} ({risk:.2f})<br>Community: {comm}<br>Connections: {degree:.0f}")
            node_color.append(risk)
            node_size.append(max(6, min(degree * 1.8 + 5, 28)))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(
                size=node_size, color=node_color,
                colorscale=RISK_SCALE,
                colorbar=dict(title=dict(text="Risk", font=dict(color=TEXT3, size=11)),
                    tickfont=dict(color=TEXT3, size=10), thickness=10, len=0.3, x=1.02),
                line=dict(width=1, color=BG),
            ),
            text=[n if G.degree(n, weight="weight") > 3 else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(family="Lato, sans-serif", size=8, color=TEXT3),
            hovertext=node_text, hoverinfo="text",
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG2,
            margin=dict(l=0, r=0, t=8, b=0), height=520, showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linewidth=0),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linewidth=0),
        )
        st.plotly_chart(fig_net, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        net1, net2 = st.columns([2, 1])

        with net1:
            st.markdown('<p class="section-header">Detected Communities</p>', unsafe_allow_html=True)
            for comm in sorted(communities, key=lambda c: c["size"], reverse=True)[:8]:
                members = comm["members"]
                comm_risk = user_risk[user_risk["community_id"] == comm["community_id"]]
                avg_risk = comm_risk["risk_score"].mean() if len(comm_risk) > 0 else 0
                high_in_comm = len(comm_risk[comm_risk["risk_level"] == "High"])
                risk_color = RED if avg_risk > 0.5 else AMBER if avg_risk > 0.2 else BLUE
                member_str = " ".join([f"@{m}" for m in members[:6]])
                if len(members) > 6:
                    member_str += f" +{len(members) - 6}"
                st.markdown(f"""
                <div class="community-card" style="border-left:4px solid {risk_color};">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                        <span style="color:{NAVY}; font-weight:700; font-size:13px;">Community {comm['community_id']}</span>
                        <span style="color:{TEXT3}; font-size:12px;">{comm['size']} members</span>
                        {"<span class='risk-badge critical'>"+str(high_in_comm)+" HIGH</span>" if high_in_comm > 0 else ""}
                        <span style="color:{risk_color}; font-weight:700; font-size:13px; margin-left:auto;">{avg_risk:.2f}</span>
                    </div>
                    <span style="color:{TEXT3}; font-size:11px;">{member_str}</span>
                </div>""", unsafe_allow_html=True)

        with net2:
            st.markdown('<p class="section-header">Network Metrics</p>', unsafe_allow_html=True)
            density = nx.density(G)
            avg_degree = sum(dict(G.degree(weight="weight")).values()) / max(len(G.nodes()), 1)
            st.markdown(f"""
            <div class="stat-block">
                <div class="stat-row"><span class="stat-label">Nodes</span><span class="stat-value">{len(G.nodes())}</span></div>
                <div class="stat-row"><span class="stat-label">Edges</span><span class="stat-value">{len(G.edges())}</span></div>
                <div class="stat-row"><span class="stat-label">Communities</span><span class="stat-value">{len(communities)}</span></div>
                <div class="stat-row"><span class="stat-label">Density</span><span class="stat-value">{density:.4f}</span></div>
                <div class="stat-row"><span class="stat-label">Avg Degree</span><span class="stat-value">{avg_degree:.1f}</span></div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  USER INTEL
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "User Intel":
    st.markdown('<p class="section-header">User Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Risk assessment and entity profiles</p>', unsafe_allow_html=True)

    with st.spinner("Computing risk profiles..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(user_risk) == 0:
        st.warning("No user data available.")
    else:
        disp_ur = user_risk.copy()
        disp_ur.columns = ["USER", "POSTS", "FLAGGED", "HIGH", "AVG", "MAX", "DEGREE", "BETW", "RISK", "LEVEL", "COMM"]

        def color_risk(val):
            return {
                "High": f"color: {RED}; font-weight: 700",
                "Medium": f"color: {AMBER}; font-weight: 600",
                "Low": f"color: {BLUE}",
            }.get(val, "")

        styled_ur = disp_ur.style.map(color_risk, subset=["LEVEL"])
        st.dataframe(styled_ur, use_container_width=True, height=300)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Intelligence Cards — High-Risk Profiles</p>', unsafe_allow_html=True)

        top_risk = user_risk[user_risk["risk_level"].isin(["High", "Medium"])].head(6)
        card_cols = st.columns(2)

        for idx, (_, u) in enumerate(top_risk.iterrows()):
            risk_99 = int(u["risk_score"] * 99)
            badge_cls = "critical" if u["risk_level"] == "High" else "medium"
            user_posts = df[df["user"] == u["user"]]
            first_seen = user_posts["timestamp"].min().strftime("%Y-%m-%d")
            flagged_pct = (u["flagged_posts"] / max(u["post_count"], 1) * 100)

            with card_cols[idx % 2]:
                st.markdown(f"""
                <div class="intel-card">
                    <div class="intel-card-header">
                        <div>
                            <div class="intel-card-name">@{u['user']}</div>
                            <div style="color:{TEXT3}; font-size:12px; margin-top:2px;">Community {u['community_id']}</div>
                        </div>
                        <div class="risk-score-badge {badge_cls}">{risk_99}</div>
                    </div>
                    <div class="intel-row"><span class="intel-label">First seen</span><span class="intel-value">{first_seen}</span></div>
                    <div class="intel-row"><span class="intel-label">Total posts</span><span class="intel-value">{u['post_count']}</span></div>
                    <div class="intel-row"><span class="intel-label">Flagged</span><span class="intel-value">{u['flagged_posts']} ({flagged_pct:.1f}%)</span></div>
                    <div class="intel-row"><span class="intel-label">High-threat</span><span class="intel-value" style="color:{RED if u['high_threat_posts']>0 else NAVY};">{u['high_threat_posts']}</span></div>
                    <div class="intel-row"><span class="intel-label">Avg score</span><span class="intel-value">{u['avg_threat_score']:.3f}</span></div>
                    <div class="intel-row"><span class="intel-label">Network degree</span><span class="intel-value">{u['network_degree']:.1f}</span></div>
                    <div class="intel-row"><span class="intel-label">Betweenness</span><span class="intel-value">{u['betweenness']:.4f}</span></div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_risk = px.histogram(user_risk, x="risk_score", nbins=25,
                color_discrete_sequence=["#111111"], template="plotly_white", labels={"risk_score": ""})
            fig_risk.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False, yaxis_title="")
            st.plotly_chart(fig_risk, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rc2:
            st.markdown('<p class="section-header">Risk Levels</p>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            rl_counts = user_risk["risk_level"].value_counts()
            fig_rl = go.Figure(data=[go.Pie(
                labels=rl_counts.index, values=rl_counts.values, hole=0.6,
                marker_colors=[THREAT_COLORS.get(l, TEXT4) for l in rl_counts.index],
                textfont=dict(family="Lato, sans-serif", size=11, color=TEXT2),
            )])
            fig_rl.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=8, b=0), height=200,
                legend=dict(font=dict(color=TEXT2, size=11), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_rl, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        csv_ur = disp_ur.to_csv(index=False)
        st.download_button(label="Export user intel (.csv)", data=csv_ur,
            file_name="socmint_user_intel.csv", mime="text/csv", use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="font-family: Lato, sans-serif; font-size: 11px; color: {TEXT3}; text-align: center;">'
    'SOCMINT Analyzer v1.0.0 · NLP + Network Analysis Engine · Built by Kimora Taylor'
    '</p>', unsafe_allow_html=True,
)
