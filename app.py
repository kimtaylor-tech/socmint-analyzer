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

# ── DESIGN TOKENS ────────────────────────────────────────────────────────────
BG       = "#0a0a0f"
SURFACE  = "#111318"
SURFACE2 = "#1a1d24"
SURFACE3 = "#22262e"
BORDER   = "#1e2028"
BORDER2  = "#2a2d36"
BORDER3  = "#3a3d46"
TEXT1    = "#f0f0f5"
TEXT2    = "#9499a8"
TEXT3    = "#5c6170"
TEXT4    = "#3a3d46"
ACCENT   = "#3b82f6"
RED      = "#ef4444"
ORANGE   = "#f97316"
AMBER    = "#eab308"
GREEN    = "#22c55e"
CYAN     = "#06b6d4"
VIOLET   = "#8b5cf6"

THREAT_COLORS = {"High": RED, "Medium": AMBER, "Low": ACCENT, "None": TEXT4}
RISK_SCALE = [[0, SURFACE2], [0.3, ACCENT], [0.6, AMBER], [1, RED]]
SERIES = [ACCENT, VIOLET, CYAN, "#10b981", AMBER, "#ec4899", ORANGE]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Inter, sans-serif", color=TEXT2, size=11),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(gridcolor=BORDER, tickfont=dict(family="Geist Mono, monospace", color=TEXT3, size=10), linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, tickfont=dict(family="Geist Mono, monospace", color=TEXT3, size=10), linecolor=BORDER),
    legend=dict(font=dict(color=TEXT2, size=10), bgcolor="rgba(0,0,0,0)"),
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT2};
  }}
  .main {{ background: {BG}; }}
  h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT1} !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em;
  }}

  /* ── Command Bar ── */
  .command-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 24px;
  }}
  .command-bar-left {{
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  .command-bar-logo {{
    font-size: 20px;
    color: {ACCENT};
  }}
  .command-bar-title {{
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: {TEXT1};
    letter-spacing: -0.02em;
  }}
  .command-bar-subtitle {{
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    color: {TEXT3};
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .command-bar-right {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}

  /* ── Live Indicator ── */
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
  }}
  .live-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: {GREEN};
    background: rgba(34, 197, 94, 0.08);
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid rgba(34, 197, 94, 0.15);
  }}
  .live-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {GREEN};
    animation: pulse 2s ease-in-out infinite;
  }}

  /* ── Section Headers ── */
  .section-header {{
    font-family: 'Geist Mono', monospace;
    color: {TEXT3};
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 10px;
    margin-bottom: 16px;
  }}

  /* ── Metric Cards ── */
  .metric-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
    text-align: left;
  }}
  .metric-card.accent-red    {{ border-left: 3px solid {RED}; }}
  .metric-card.accent-amber  {{ border-left: 3px solid {AMBER}; }}
  .metric-card.accent-blue   {{ border-left: 3px solid {ACCENT}; }}
  .metric-card.accent-green  {{ border-left: 3px solid {GREEN}; }}

  .metric-label {{
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT3};
    margin-bottom: 6px;
  }}
  .metric-value {{
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 600;
    color: {TEXT1};
    letter-spacing: -0.02em;
    line-height: 1;
  }}
  .metric-delta {{
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    margin-top: 8px;
    color: {TEXT3};
  }}
  .metric-delta.danger  {{ color: {RED}; }}
  .metric-delta.warning {{ color: {AMBER}; }}
  .metric-delta.success {{ color: {GREEN}; }}
  .metric-delta.info    {{ color: {ACCENT}; }}

  /* ── Risk Badges ── */
  .risk-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .risk-badge.critical {{ background: rgba(239,68,68,0.12); color: {RED}; }}
  .risk-badge.high     {{ background: rgba(249,115,22,0.12); color: {ORANGE}; }}
  .risk-badge.medium   {{ background: rgba(234,179,8,0.12);  color: {AMBER}; }}
  .risk-badge.low      {{ background: rgba(59,130,246,0.12);  color: {ACCENT}; }}

  /* ── Alert Cards (Dataminr-style) ── */
  .alert-feed {{
    max-height: 700px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: {BORDER2} transparent;
    padding-right: 4px;
  }}
  .alert-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {TEXT4};
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
    line-height: 1.55;
  }}
  .alert-card.severity-high   {{ border-left-color: {RED}; }}
  .alert-card.severity-medium {{ border-left-color: {AMBER}; }}
  .alert-card.severity-low    {{ border-left-color: {ACCENT}; }}

  .alert-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }}
  .alert-user {{
    color: {TEXT1};
    font-weight: 500;
    font-size: 13px;
  }}
  .alert-meta {{
    color: {TEXT3};
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
  }}
  .alert-score {{
    margin-left: auto;
    font-family: 'Geist Mono', monospace;
    font-size: 12px;
    font-weight: 600;
  }}
  .alert-text {{
    color: {TEXT2};
    font-size: 13px;
    margin-bottom: 10px;
    line-height: 1.5;
  }}
  .alert-tags {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    color: {TEXT3};
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
  }}

  /* ── Community Cards ── */
  .community-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
  }}

  /* ── Stat Block ── */
  .stat-block {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px 24px;
    line-height: 2.2;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2px 0;
    border-bottom: 1px solid {BORDER};
  }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-label {{ color: {TEXT3}; font-family: 'Geist Mono', monospace; font-size: 10px; letter-spacing: 0.05em; text-transform: uppercase; }}
  .stat-value {{ color: {TEXT1}; font-family: 'Geist Mono', monospace; font-size: 13px; font-weight: 500; }}

  /* ── Intelligence Card (Recorded Future style) ── */
  .intel-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px 24px;
    margin: 12px 0;
  }}
  .intel-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid {BORDER};
  }}
  .intel-card-name {{
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: {TEXT1};
  }}
  .risk-score-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 700;
  }}
  .risk-score-badge.critical {{ background: rgba(239,68,68,0.15); color: {RED}; }}
  .risk-score-badge.high     {{ background: rgba(249,115,22,0.15); color: {ORANGE}; }}
  .risk-score-badge.medium   {{ background: rgba(234,179,8,0.15);  color: {AMBER}; }}
  .risk-score-badge.low      {{ background: rgba(59,130,246,0.15);  color: {ACCENT}; }}

  .intel-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 12px;
  }}
  .intel-label {{ color: {TEXT3}; font-family: 'Geist Mono', monospace; font-size: 11px; }}
  .intel-value {{ color: {TEXT1}; font-family: 'Geist Mono', monospace; font-size: 11px; }}

  /* ── Sidebar ── */
  .stSidebar {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
  }}

  /* ── Buttons ── */
  .stButton button {{
    background: {TEXT1};
    color: {BG};
    border: none;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 12px;
    padding: 8px 16px;
  }}
  .stButton button:hover {{ background: #d4d4d8; }}

  /* ── Form Labels ── */
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stSelectbox"] label,
  div[data-testid="stRadio"] label {{
    color: {TEXT3};
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    font-family: 'Geist Mono', monospace;
  }}

  .stDataFrame {{ font-family: 'Geist Mono', monospace; font-size: 11px; }}

  hr {{ border-color: {BORDER} !important; opacity: 0.5; }}
  .stRadio > div {{ gap: 2px; }}

  .stDownloadButton button {{
    background: transparent;
    color: {TEXT2};
    border: 1px solid {BORDER2};
    border-radius: 6px;
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.03em;
  }}
  .stDownloadButton button:hover {{ border-color: {TEXT3}; color: {TEXT1}; }}

  /* ── Filter Chips ── */
  .filter-chips {{
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .filter-chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.03em;
    border: 1px solid {BORDER2};
    color: {TEXT2};
    background: transparent;
  }}
  .filter-chip.active {{ background: rgba(59,130,246,0.1); border-color: {ACCENT}; color: {ACCENT}; }}

  /* ── Timestamp ── */
  .ts {{
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    color: {TEXT4};
  }}
</style>
""", unsafe_allow_html=True)

# ── COMMAND BAR ──────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="command-bar">
    <div class="command-bar-left">
        <span class="command-bar-logo">◉</span>
        <div>
            <div class="command-bar-title">SOCMINT Analyzer</div>
            <div class="command-bar-subtitle">Social Media Intelligence Platform</div>
        </div>
    </div>
    <div class="command-bar-right">
        <span class="ts">Last scan: {now_str}</span>
        <div class="live-badge">
            <div class="live-dot"></div>
            LIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">CONTROLS</p>', unsafe_allow_html=True)

    n_posts = st.slider("POST VOLUME", 500, 5000, 2000, step=250)

    if st.button("Ingest & Analyze", use_container_width=True):
        st.session_state["run_seed"] = int(time.time())
        st.cache_data.clear()

    st.markdown("---")
    st.markdown('<p class="section-header">FILTERS</p>', unsafe_allow_html=True)

    threat_filter = st.multiselect(
        "THREAT LEVEL",
        ["High", "Medium", "Low", "None"],
        default=["High", "Medium", "Low", "None"],
    )

    st.markdown("---")
    st.markdown('<p class="section-header">VIEW</p>', unsafe_allow_html=True)

    view_mode = st.radio(
        "DASHBOARD VIEW",
        ["Overview", "Alert Feed", "Network", "User Intel"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-family: Geist Mono, monospace; font-size: 10px; color: {TEXT4}; line-height: 1.8;">
        <div class="section-header" style="margin-bottom:10px;">SYSTEM</div>
        <div>Status: <span style="color:{GREEN};">operational</span></div>
        <div>Engine: NLP + NetworkX</div>
        <div>Version: 1.0.0</div>
        <div style="margin-top:8px; color:{TEXT4};">Kimora Taylor</div>
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

with st.spinner("Analyzing..."):
    df = run_pipeline(n_posts, current_seed)

filtered = df[df["threat_level"].isin(threat_filter)]
flagged = filtered[filtered["threat_level"] != "None"]

# ── HELPER: Risk badge HTML ──────────────────────────────────────────────────
def risk_badge(level):
    cls = level.lower() if level in ("High", "Medium", "Low") else "low"
    if level == "High":
        cls = "critical"
    return f'<span class="risk-badge {cls}">{level}</span>'

def risk_score_color(score):
    if score >= 0.6: return RED
    if score >= 0.3: return AMBER
    return ACCENT

# ── KPI STRIP ────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

total = len(filtered)
flagged_count = len(flagged)
high_count = len(filtered[filtered["threat_level"] == "High"])
med_count = len(filtered[filtered["threat_level"] == "Medium"])
unique_flagged_users = flagged["user"].nunique()
pct = (flagged_count / total * 100) if total else 0

with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Posts Analyzed</div>
        <div class="metric-value">{total:,}</div>
        <div class="metric-delta info">{n_posts:,} ingested</div>
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
        <div class="metric-delta danger">{"requires review" if high_count > 0 else "clear"}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Medium Threat</div>
        <div class="metric-value">{med_count}</div>
        <div class="metric-delta warning">monitoring</div>
    </div>""", unsafe_allow_html=True)
with m5:
    st.markdown(f"""<div class="metric-card accent-blue">
        <div class="metric-label">Unique Accounts</div>
        <div class="metric-value">{unique_flagged_users}</div>
        <div class="metric-delta info">flagged users</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═════════════════���════════════════════════════════════════════════════════════
if view_mode == "Overview":
    # Primary visualization — full-width timeline
    st.markdown('<p class="section-header">THREAT ACTIVITY — 7 DAY TIMELINE</p>', unsafe_allow_html=True)
    tl = filtered.copy()
    tl["day_hour"] = tl["timestamp"].dt.floor("6h")
    timeline = tl.groupby(["day_hour", "threat_level"]).size().reset_index(name="count")
    fig_tl = px.area(
        timeline, x="day_hour", y="count", color="threat_level",
        color_discrete_map=THREAT_COLORS, template="plotly_dark",
        labels={"day_hour": "", "count": "", "threat_level": ""},
    )
    fig_tl.update_layout(**PLOTLY_LAYOUT, height=220)
    fig_tl.update_traces(line=dict(width=1.5))
    st.plotly_chart(fig_tl, use_container_width=True)

    # Secondary panels — 3 columns
    ov1, ov2, ov3 = st.columns(3)

    with ov1:
        st.markdown('<p class="section-header">THREAT BREAKDOWN</p>', unsafe_allow_html=True)
        tc = flagged["threat_level"].value_counts()
        if len(tc) > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=tc.index, values=tc.values, hole=0.65,
                marker_colors=[THREAT_COLORS.get(l, TEXT4) for l in tc.index],
                textfont=dict(family="Geist Mono, monospace", size=10, color=TEXT2),
                hovertemplate="<b>%{label}</b><br>%{value} posts<br>%{percent}<extra></extra>",
            )])
            fig_pie.update_layout(paper_bgcolor=BG, margin=dict(l=0, r=0, t=8, b=0), height=260,
                legend=dict(font=dict(color=TEXT2, size=10), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pie, use_container_width=True)

    with ov2:
        st.markdown('<p class="section-header">SENTIMENT ANALYSIS</p>', unsafe_allow_html=True)
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Histogram(
            x=filtered[filtered["threat_level"] == "None"]["sentiment_polarity"],
            name="Benign", marker_color=TEXT4, opacity=0.4, nbinsx=30,
        ))
        fig_sent.add_trace(go.Histogram(
            x=flagged["sentiment_polarity"],
            name="Flagged", marker_color=RED, opacity=0.7, nbinsx=30,
        ))
        fig_sent.add_vline(x=0, line_dash="dot", line_color=BORDER3, opacity=0.5)
        fig_sent.update_layout(**PLOTLY_LAYOUT, height=260, barmode="overlay",
            xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_sent, use_container_width=True)

    with ov3:
        st.markdown('<p class="section-header">TOP FLAGGED ACCOUNTS</p>', unsafe_allow_html=True)
        top_users = flagged["user"].value_counts().head(8).reset_index()
        top_users.columns = ["user", "count"]
        if len(top_users) > 0:
            fig_tu = px.bar(
                top_users, x="count", y="user", orientation="h",
                color="count", color_continuous_scale=[[0, SURFACE2], [1, RED]],
                template="plotly_dark", labels={"count": "", "user": ""},
            )
            fig_tu.update_layout(**PLOTLY_LAYOUT, height=260, coloraxis_showscale=False)
            fig_tu.update_layout(yaxis=dict(tickfont=dict(family="Geist Mono, monospace", color=TEXT2, size=10)))
            st.plotly_chart(fig_tu, use_container_width=True)

    # Bottom row — hashtags and recent high-threat
    bt1, bt2 = st.columns([1, 1.5])

    with bt1:
        st.markdown('<p class="section-header">TRENDING SIGNALS</p>', unsafe_allow_html=True)
        all_tags = " ".join(flagged["hashtags"].dropna().astype(str)).split()
        all_tags = [t for t in all_tags if t.startswith("#")]
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(10).reset_index()
            tag_counts.columns = ["hashtag", "count"]
            fig_tags = px.bar(
                tag_counts, x="count", y="hashtag", orientation="h",
                color="count", color_continuous_scale=[[0, SURFACE2], [1, AMBER]],
                template="plotly_dark", labels={"count": "", "hashtag": ""},
            )
            fig_tags.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
            fig_tags.update_layout(yaxis=dict(tickfont=dict(family="Geist Mono, monospace", color=TEXT2, size=10)))
            st.plotly_chart(fig_tags, use_container_width=True)

    with bt2:
        st.markdown('<p class="section-header">RECENT HIGH-THREAT ALERTS</p>', unsafe_allow_html=True)
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
                        <span>{row['likes']} likes</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:{TEXT3}; font-size:12px;">No high-threat alerts in current data.</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT FEED (Dataminr Pulse-inspired)
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Alert Feed":
    st.markdown('<p class="section-header">ALERT FEED — REAL-TIME THREAT STREAM</p>', unsafe_allow_html=True)

    # Filter chips
    chip_html = '<div class="filter-chips">'
    counts = {"High": high_count, "Medium": med_count, "Low": len(filtered[filtered["threat_level"] == "Low"])}
    for lvl, cnt in counts.items():
        active = "active" if lvl in threat_filter else ""
        chip_html += f'<span class="filter-chip {active}">{lvl} ({cnt})</span>'
    chip_html += f'<span class="filter-chip" style="margin-left:auto;">Total: {flagged_count}</span>'
    chip_html += '</div>'
    st.markdown(chip_html, unsafe_allow_html=True)

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
        st.markdown('<p class="section-header">THREAT SUMMARY</p>', unsafe_allow_html=True)

        # Severity distribution mini chart
        sev_order = ["High", "Medium", "Low"]
        sev_vals = [len(flagged[flagged["threat_level"] == s]) for s in sev_order]
        fig_sev = go.Figure(go.Bar(
            x=sev_vals, y=sev_order, orientation="h",
            marker_color=[THREAT_COLORS[s] for s in sev_order],
            text=sev_vals, textposition="outside",
            textfont=dict(family="Geist Mono, monospace", color=TEXT2, size=10),
        ))
        fig_sev.update_layout(**PLOTLY_LAYOUT, height=140)
        fig_sev.update_layout(yaxis=dict(tickfont=dict(family="Geist Mono, monospace", color=TEXT2, size=10)))
        st.plotly_chart(fig_sev, use_container_width=True)

        # Top keywords
        st.markdown('<p class="section-header">DETECTED KEYWORDS</p>', unsafe_allow_html=True)
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
                <div style="display:flex; align-items:center; gap:8px; margin:4px 0; font-family:Geist Mono,monospace; font-size:11px;">
                    <span style="color:{TEXT2}; min-width:90px;">{kw}</span>
                    <div style="flex:1; background:{BORDER}; height:4px; border-radius:2px;">
                        <div style="width:{bar_w}%; background:{RED}; height:4px; border-radius:2px;"></div>
                    </div>
                    <span style="color:{TEXT3}; min-width:24px; text-align:right;">{cnt}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        csv = flagged[[
            "timestamp", "user", "text", "threat_level", "threat_score",
            "sentiment_polarity", "high_keywords", "coordination_patterns",
        ]].to_csv(index=False)
        st.download_button(
            label="Export alerts (.csv)",
            data=csv, file_name="socmint_alerts.csv",
            mime="text/csv", use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK (Palantir/Maltego-inspired)
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Network":
    st.markdown('<p class="section-header">INTERACTION NETWORK — COMMUNITY DETECTION</p>', unsafe_allow_html=True)

    with st.spinner("Building network..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(G.nodes()) == 0:
        st.warning("No network data available.")
    else:
        edge_x, edge_y = [], []
        edge_weights = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(d.get("weight", 1))

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.3, color="rgba(42,45,54,0.6)"), hoverinfo="none",
        )

        # Color nodes by community using series palette
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

            node_text.append(
                f"<b>@{node}</b><br>"
                f"Risk: {risk_lvl} ({risk:.2f})<br>"
                f"Community: {comm}<br>"
                f"Connections: {degree:.0f}"
            )
            node_color.append(risk)
            node_size.append(max(5, min(degree * 1.8 + 4, 26)))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(
                size=node_size, color=node_color,
                colorscale=RISK_SCALE,
                colorbar=dict(
                    title=dict(text="RISK", font=dict(color=TEXT3, size=10, family="Geist Mono, monospace")),
                    tickfont=dict(color=TEXT3, size=9),
                    thickness=10, len=0.3, x=1.02,
                ),
                line=dict(width=0.5, color=BG),
            ),
            text=[n if G.degree(n, weight="weight") > 3 else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(family="Geist Mono, monospace", size=7, color=TEXT3),
            hovertext=node_text, hoverinfo="text",
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            margin=dict(l=0, r=0, t=8, b=0), height=520, showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linewidth=0),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linewidth=0),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Community + stats row
        net1, net2 = st.columns([2, 1])

        with net1:
            st.markdown('<p class="section-header">DETECTED COMMUNITIES</p>', unsafe_allow_html=True)
            for comm in sorted(communities, key=lambda c: c["size"], reverse=True)[:8]:
                members = comm["members"]
                comm_risk = user_risk[user_risk["community_id"] == comm["community_id"]]
                avg_risk = comm_risk["risk_score"].mean() if len(comm_risk) > 0 else 0
                high_in_comm = len(comm_risk[comm_risk["risk_level"] == "High"])
                risk_color = RED if avg_risk > 0.5 else AMBER if avg_risk > 0.2 else ACCENT
                member_str = " ".join([f"@{m}" for m in members[:6]])
                if len(members) > 6:
                    member_str += f" +{len(members) - 6}"
                st.markdown(f"""
                <div class="community-card" style="border-left:2px solid {risk_color};">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                        <span style="color:{TEXT1}; font-weight:600; font-size:12px;">Community {comm['community_id']}</span>
                        <span style="color:{TEXT3}; font-family:Geist Mono,monospace; font-size:10px;">{comm['size']} members</span>
                        {"<span class='risk-badge critical'>"+str(high_in_comm)+" HIGH</span>" if high_in_comm > 0 else ""}
                        <span style="color:{risk_color}; font-family:Geist Mono,monospace; font-size:10px; margin-left:auto;">{avg_risk:.2f}</span>
                    </div>
                    <span style="color:{TEXT3}; font-family:Geist Mono,monospace; font-size:10px;">{member_str}</span>
                </div>""", unsafe_allow_html=True)

        with net2:
            st.markdown('<p class="section-header">NETWORK METRICS</p>', unsafe_allow_html=True)
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
#  USER INTEL (Recorded Future Intelligence Cards)
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "User Intel":
    st.markdown('<p class="section-header">USER INTELLIGENCE — RISK ASSESSMENT</p>', unsafe_allow_html=True)

    with st.spinner("Computing risk profiles..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(user_risk) == 0:
        st.warning("No user data available.")
    else:
        # Data table
        disp_ur = user_risk.copy()
        disp_ur.columns = [
            "USER", "POSTS", "FLAGGED", "HIGH", "AVG",
            "MAX", "DEGREE", "BETW", "RISK",
            "LEVEL", "COMM",
        ]

        def color_risk(val):
            return {
                "High": f"color: {RED}; font-weight: 600",
                "Medium": f"color: {AMBER}; font-weight: 500",
                "Low": f"color: {ACCENT}",
            }.get(val, "")

        styled_ur = disp_ur.style.map(color_risk, subset=["LEVEL"])
        st.dataframe(styled_ur, use_container_width=True, height=350)

        # Intelligence Cards for top risk users
        st.markdown('<p class="section-header">HIGH-RISK INTELLIGENCE CARDS</p>', unsafe_allow_html=True)

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
                            <div style="color:{TEXT3}; font-family:Geist Mono,monospace; font-size:10px; margin-top:2px;">Community {u['community_id']}</div>
                        </div>
                        <div class="risk-score-badge {badge_cls}">{risk_99}</div>
                    </div>
                    <div class="intel-row"><span class="intel-label">First seen</span><span class="intel-value">{first_seen}</span></div>
                    <div class="intel-row"><span class="intel-label">Total posts</span><span class="intel-value">{u['post_count']}</span></div>
                    <div class="intel-row"><span class="intel-label">Flagged</span><span class="intel-value">{u['flagged_posts']} ({flagged_pct:.1f}%)</span></div>
                    <div class="intel-row"><span class="intel-label">High-threat</span><span class="intel-value" style="color:{RED if u['high_threat_posts']>0 else TEXT1};">{u['high_threat_posts']}</span></div>
                    <div class="intel-row"><span class="intel-label">Avg score</span><span class="intel-value">{u['avg_threat_score']:.3f}</span></div>
                    <div class="intel-row"><span class="intel-label">Network degree</span><span class="intel-value">{u['network_degree']:.1f}</span></div>
                    <div class="intel-row"><span class="intel-label">Betweenness</span><span class="intel-value">{u['betweenness']:.4f}</span></div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Risk distribution charts
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('<p class="section-header">RISK DISTRIBUTION</p>', unsafe_allow_html=True)
            fig_risk = px.histogram(
                user_risk, x="risk_score", nbins=25,
                color_discrete_sequence=[ACCENT], template="plotly_dark",
                labels={"risk_score": ""},
            )
            fig_risk.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False, yaxis_title="")
            st.plotly_chart(fig_risk, use_container_width=True)

        with rc2:
            st.markdown('<p class="section-header">RISK LEVELS</p>', unsafe_allow_html=True)
            rl_counts = user_risk["risk_level"].value_counts()
            fig_rl = go.Figure(data=[go.Pie(
                labels=rl_counts.index, values=rl_counts.values, hole=0.6,
                marker_colors=[THREAT_COLORS.get(l, TEXT4) for l in rl_counts.index],
                textfont=dict(family="Geist Mono, monospace", size=10, color=TEXT2),
            )])
            fig_rl.update_layout(paper_bgcolor=BG, margin=dict(l=0, r=0, t=8, b=0), height=200,
                legend=dict(font=dict(color=TEXT2, size=10), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_rl, use_container_width=True)

        csv_ur = disp_ur.to_csv(index=False)
        st.download_button(
            label="Export user intel (.csv)",
            data=csv_ur, file_name="socmint_user_intel.csv",
            mime="text/csv", use_container_width=True,
        )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="font-family: Geist Mono, monospace; font-size: 9px; color: {TEXT4}; text-align: center; letter-spacing:0.08em;">'
    'SOCMINT ANALYZER v1.0.0 · NLP + NETWORK ANALYSIS ENGINE · KIMORA TAYLOR'
    '</p>', unsafe_allow_html=True,
)
