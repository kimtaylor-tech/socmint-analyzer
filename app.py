import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time
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
BG       = "#09090b"
SURFACE  = "#111113"
SURFACE2 = "#18181b"
BORDER   = "#1f1f23"
BORDER2  = "#27272a"
TEXT1    = "#fafafa"
TEXT2    = "#a1a1aa"
TEXT3    = "#71717a"
TEXT4    = "#3f3f46"
ACCENT   = "#3b82f6"
RED      = "#ef4444"
AMBER    = "#f59e0b"
GREEN    = "#22c55e"

THREAT_COLORS = {"High": RED, "Medium": AMBER, "Low": ACCENT, "None": "#27272a"}
RISK_SCALE = [[0, "#18181b"], [0.3, ACCENT], [0.6, AMBER], [1, RED]]
SERIES = [ACCENT, "#8b5cf6", "#ec4899", AMBER, GREEN, "#06b6d4", "#f97316"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Inter, sans-serif", color=TEXT2, size=11),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(gridcolor=BORDER, tickfont=dict(family="Geist Mono, monospace", color=TEXT3, size=10)),
    yaxis=dict(gridcolor=BORDER, tickfont=dict(family="Geist Mono, monospace", color=TEXT3, size=10)),
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
    font-family: 'Inter', sans-serif;
    color: {TEXT1};
    font-weight: 600;
    letter-spacing: -0.025em;
  }}

  .section-header {{
    font-family: 'Geist Mono', monospace;
    color: {TEXT3};
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 12px;
    margin-bottom: 20px;
  }}

  .metric-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    text-align: left;
  }}
  .metric-label {{
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT3};
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 600;
    color: {TEXT1};
    letter-spacing: -0.02em;
    line-height: 1;
  }}
  .metric-delta {{
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    margin-top: 6px;
    color: {TEXT3};
  }}
  .metric-delta.danger  {{ color: {RED}; }}
  .metric-delta.warning {{ color: {AMBER}; }}
  .metric-delta.success {{ color: {GREEN}; }}
  .metric-delta.info    {{ color: {ACCENT}; }}

  .post-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 2px solid {RED};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 13px;
    line-height: 1.6;
  }}
  .post-card.medium {{ border-left-color: {AMBER}; }}
  .post-card.low    {{ border-left-color: {ACCENT}; }}

  .community-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
  }}

  .stat-block {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    line-height: 2;
  }}
  .stat-label {{ color: {TEXT3}; font-family: 'Geist Mono', monospace; font-size: 11px; }}
  .stat-value {{ color: {TEXT1}; font-weight: 600; }}

  .stSidebar {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
  }}

  .stButton button {{
    background: {TEXT1};
    color: {BG};
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 13px;
    letter-spacing: 0;
    padding: 10px 16px;
  }}
  .stButton button:hover {{ background: #e4e4e7; }}

  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stSelectbox"] label,
  div[data-testid="stRadio"] label {{
    color: {TEXT3};
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    font-family: 'Geist Mono', monospace;
  }}

  .stDataFrame {{ font-family: 'Geist Mono', monospace; font-size: 12px; }}

  /* Clean dividers */
  hr {{ border-color: {BORDER} !important; opacity: 0.5; }}

  /* Streamlit overrides for dark theme consistency */
  .stRadio > div {{ gap: 4px; }}
  .stDownloadButton button {{
    background: transparent;
    color: {TEXT2};
    border: 1px solid {BORDER2};
    border-radius: 8px;
    font-family: 'Geist Mono', monospace;
    font-size: 12px;
  }}
  .stDownloadButton button:hover {{ border-color: {TEXT3}; color: {TEXT1}; }}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────
hc1, hc2, hc3 = st.columns([0.5, 5, 1])
with hc1:
    st.markdown(f'<p style="font-size:28px; margin-top:8px;">◉</p>', unsafe_allow_html=True)
with hc2:
    st.markdown("# SOCMINT Analyzer")
    st.markdown(f'<p style="color:{TEXT3}; font-family: Geist Mono, monospace; font-size:12px; letter-spacing:0.05em; margin-top:-10px;">Social Media Intelligence Platform</p>', unsafe_allow_html=True)
with hc3:
    st.markdown(f'<p style="color:{GREEN}; font-family: Geist Mono, monospace; font-size:11px; margin-top:16px; text-align:right;">● Online</p>', unsafe_allow_html=True)

st.markdown("---")

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
        ["Overview", "Flagged Posts", "Network Graph", "User Risk"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f'<p style="font-family: Geist Mono, monospace; font-size: 10px; color: {TEXT4};">SOCMINT v1.0.0 · Kimora Taylor</p>', unsafe_allow_html=True)

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

# ── METRICS ──────────────────────────────────────────────────────────────────
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
        <div class="metric-delta info">ingested feed</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Flagged</div>
        <div class="metric-value">{flagged_count:,}</div>
        <div class="metric-delta warning">{pct:.1f}% of total</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">High Threat</div>
        <div class="metric-value">{high_count}</div>
        <div class="metric-delta danger">{"immediate review" if high_count > 0 else "clear"}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Medium Threat</div>
        <div class="metric-value">{med_count}</div>
        <div class="metric-delta warning">monitoring</div>
    </div>""", unsafe_allow_html=True)
with m5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Flagged Users</div>
        <div class="metric-value">{unique_flagged_users}</div>
        <div class="metric-delta info">unique accounts</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Overview":
    ov1, ov2, ov3 = st.columns([2, 1.5, 1.5])

    with ov1:
        st.markdown('<p class="section-header">THREAT ACTIVITY — 7 DAY TIMELINE</p>', unsafe_allow_html=True)
        tl = filtered.copy()
        tl["day_hour"] = tl["timestamp"].dt.floor("6h")
        timeline = tl.groupby(["day_hour", "threat_level"]).size().reset_index(name="count")
        fig_tl = px.bar(
            timeline, x="day_hour", y="count", color="threat_level",
            color_discrete_map=THREAT_COLORS, template="plotly_dark",
            labels={"day_hour": "", "count": "", "threat_level": "Level"},
        )
        fig_tl.update_layout(**PLOTLY_LAYOUT, bargap=0.2, height=280)
        st.plotly_chart(fig_tl, use_container_width=True)

    with ov2:
        st.markdown('<p class="section-header">THREAT BREAKDOWN</p>', unsafe_allow_html=True)
        tc = flagged["threat_level"].value_counts()
        if len(tc) > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=tc.index, values=tc.values, hole=0.65,
                marker_colors=[THREAT_COLORS.get(l, TEXT4) for l in tc.index],
                textfont=dict(family="Geist Mono, monospace", size=10, color=TEXT2),
                hovertemplate="<b>%{label}</b><br>%{value} posts<br>%{percent}<extra></extra>",
            )])
            fig_pie.update_layout(paper_bgcolor=BG, margin=dict(l=0, r=0, t=8, b=0), height=280,
                legend=dict(font=dict(color=TEXT2, size=10), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No flagged posts with current filters.")

    with ov3:
        st.markdown('<p class="section-header">SENTIMENT DISTRIBUTION</p>', unsafe_allow_html=True)
        fig_sent = px.histogram(
            filtered, x="sentiment_polarity", nbins=40,
            color_discrete_sequence=[ACCENT], template="plotly_dark",
            labels={"sentiment_polarity": "← Negative · Positive →"},
        )
        fig_sent.add_vline(x=0, line_dash="dot", line_color=RED, opacity=0.3)
        fig_sent.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        fig_sent.update_layout(yaxis_title="")
        st.plotly_chart(fig_sent, use_container_width=True)

    # Second row
    ov4, ov5 = st.columns(2)

    with ov4:
        st.markdown('<p class="section-header">TOP FLAGGED USERS</p>', unsafe_allow_html=True)
        top_users = flagged["user"].value_counts().head(12).reset_index()
        top_users.columns = ["user", "flagged_posts"]
        if len(top_users) > 0:
            fig_tu = px.bar(
                top_users, x="flagged_posts", y="user", orientation="h",
                color="flagged_posts", color_continuous_scale=[[0, SURFACE2], [1, RED]],
                template="plotly_dark", labels={"flagged_posts": "", "user": ""},
            )
            fig_tu.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            fig_tu.update_layout(yaxis=dict(tickfont=dict(family="Geist Mono, monospace", color=TEXT2, size=11)))
            st.plotly_chart(fig_tu, use_container_width=True)

    with ov5:
        st.markdown('<p class="section-header">TRENDING HASHTAGS IN FLAGGED POSTS</p>', unsafe_allow_html=True)
        all_tags = " ".join(flagged["hashtags"].dropna().astype(str)).split()
        all_tags = [t for t in all_tags if t.startswith("#")]
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(12).reset_index()
            tag_counts.columns = ["hashtag", "count"]
            fig_tags = px.bar(
                tag_counts, x="count", y="hashtag", orientation="h",
                color="count", color_continuous_scale=[[0, SURFACE2], [1, AMBER]],
                template="plotly_dark", labels={"count": "", "hashtag": ""},
            )
            fig_tags.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            fig_tags.update_layout(yaxis=dict(tickfont=dict(family="Geist Mono, monospace", color=TEXT2, size=11)))
            st.plotly_chart(fig_tags, use_container_width=True)
        else:
            st.info("No hashtags found in flagged posts.")


# ══════════════════════════════════════════════════════════════════════════════
#  FLAGGED POSTS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Flagged Posts":
    st.markdown('<p class="section-header">FLAGGED POSTS — BY THREAT SCORE</p>', unsafe_allow_html=True)

    if len(flagged) > 0:
        display_cols = [
            "timestamp", "user", "text", "threat_level", "threat_score",
            "sentiment_polarity", "high_keywords", "med_keywords",
            "coordination_patterns", "likes", "reposts",
        ]
        disp = flagged[display_cols].copy()
        disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        disp = disp.sort_values("threat_score", ascending=False)
        disp.columns = [
            "TIMESTAMP", "USER", "POST", "THREAT", "SCORE",
            "SENTIMENT", "HIGH KW", "MED KW", "COORD", "LIKES", "REPOSTS",
        ]

        def color_threat(val):
            return {
                "High": f"color: {RED}; font-weight: 600",
                "Medium": f"color: {AMBER}; font-weight: 500",
                "Low": f"color: {ACCENT}",
            }.get(val, "")

        styled = disp.style.map(color_threat, subset=["THREAT"])
        st.dataframe(styled, use_container_width=True, height=450)

        csv = disp.to_csv(index=False)
        st.download_button(
            label="Export flagged posts (.csv)",
            data=csv, file_name="socmint_flagged_posts.csv",
            mime="text/csv", use_container_width=True,
        )
    else:
        st.info("No posts match the current threat filters.")

    # High-threat detail cards
    high_posts = flagged[flagged["threat_level"] == "High"].sort_values("threat_score", ascending=False).head(10)
    if len(high_posts) > 0:
        st.markdown("---")
        st.markdown('<p class="section-header">HIGH-THREAT POST DETAILS</p>', unsafe_allow_html=True)
        for _, row in high_posts.iterrows():
            kw_display = row["high_keywords"]
            if row["med_keywords"]:
                kw_display += (", " if kw_display else "") + row["med_keywords"]
            st.markdown(f"""
            <div class="post-card">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                    <span style="color:{RED}; font-weight:600; font-family:Geist Mono,monospace; font-size:11px; letter-spacing:0.05em;">HIGH</span>
                    <span style="color:{TEXT2}; font-size:13px;">@{row['user']}</span>
                    <span style="color:{TEXT4}; font-family:Geist Mono,monospace; font-size:11px;">{row['timestamp'].strftime('%m/%d %H:%M')}</span>
                    <span style="color:{RED}; font-family:Geist Mono,monospace; font-size:11px; margin-left:auto;">{row['threat_score']:.2f}</span>
                </div>
                <p style="color:{TEXT1}; font-size:14px; margin:0 0 12px 0; line-height:1.5;">"{row['text']}"</p>
                <div style="color:{TEXT3}; font-family:Geist Mono,monospace; font-size:11px; display:flex; gap:16px; flex-wrap:wrap;">
                    <span>keywords: {kw_display or '—'}</span>
                    <span>sentiment: {row['sentiment_polarity']:.2f}</span>
                    <span>coord: {row['coordination_patterns']}</span>
                    <span>{row['likes']} likes · {row['reposts']} reposts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Network Graph":
    st.markdown('<p class="section-header">INTERACTION NETWORK — COMMUNITY DETECTION</p>', unsafe_allow_html=True)

    with st.spinner("Building network..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(G.nodes()) == 0:
        st.warning("No network data available.")
    else:
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.3, color=BORDER2), hoverinfo="none",
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

            node_text.append(f"@{node}<br>Risk: {risk_lvl} ({risk:.2f})<br>Community: {comm}<br>Connections: {degree:.0f}")
            node_color.append(risk)
            node_size.append(max(5, min(degree * 2 + 4, 28)))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(
                size=node_size, color=node_color,
                colorscale=RISK_SCALE,
                colorbar=dict(title=dict(text="Risk", font=dict(color=TEXT3, size=11)),
                              tickfont=dict(color=TEXT3, size=10), thickness=12, len=0.4),
                line=dict(width=0.5, color=BG),
            ),
            text=[n if G.degree(n, weight="weight") > 3 else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(family="Geist Mono, monospace", size=8, color=TEXT3),
            hovertext=node_text, hoverinfo="text",
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            margin=dict(l=0, r=0, t=8, b=0), height=550, showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Community + stats
        net1, net2 = st.columns([2, 1])

        with net1:
            st.markdown('<p class="section-header">DETECTED COMMUNITIES</p>', unsafe_allow_html=True)
            for comm in sorted(communities, key=lambda c: c["size"], reverse=True)[:8]:
                members = comm["members"]
                comm_risk = user_risk[user_risk["community_id"] == comm["community_id"]]
                avg_risk = comm_risk["risk_score"].mean() if len(comm_risk) > 0 else 0
                risk_color = RED if avg_risk > 0.5 else AMBER if avg_risk > 0.2 else ACCENT
                member_str = ", ".join([f"@{m}" for m in members[:6]])
                if len(members) > 6:
                    member_str += f" +{len(members) - 6} more"
                st.markdown(f"""
                <div class="community-card" style="border-left:2px solid {risk_color};">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                        <span style="color:{TEXT1}; font-weight:600; font-size:13px;">Community {comm['community_id']}</span>
                        <span style="color:{TEXT3}; font-family:Geist Mono,monospace; font-size:11px;">{comm['size']} members</span>
                        <span style="color:{risk_color}; font-family:Geist Mono,monospace; font-size:11px; margin-left:auto;">risk {avg_risk:.2f}</span>
                    </div>
                    <span style="color:{TEXT3}; font-family:Geist Mono,monospace; font-size:11px;">{member_str}</span>
                </div>
                """, unsafe_allow_html=True)

        with net2:
            st.markdown('<p class="section-header">NETWORK STATS</p>', unsafe_allow_html=True)
            density = nx.density(G)
            st.markdown(f"""
            <div class="stat-block">
                <div><span class="stat-label">Nodes</span> <span class="stat-value">{len(G.nodes())}</span></div>
                <div><span class="stat-label">Edges</span> <span class="stat-value">{len(G.edges())}</span></div>
                <div><span class="stat-label">Communities</span> <span class="stat-value">{len(communities)}</span></div>
                <div><span class="stat-label">Density</span> <span class="stat-value">{density:.4f}</span></div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  USER RISK
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "User Risk":
    st.markdown('<p class="section-header">USER RISK ASSESSMENT</p>', unsafe_allow_html=True)

    with st.spinner("Computing risk profiles..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(user_risk) == 0:
        st.warning("No user data available.")
    else:
        ur1, ur2 = st.columns([2, 1])

        with ur1:
            disp_ur = user_risk.copy()
            disp_ur.columns = [
                "USER", "POSTS", "FLAGGED", "HIGH", "AVG SCORE",
                "MAX SCORE", "DEGREE", "BETWEENNESS", "RISK",
                "LEVEL", "COMMUNITY",
            ]

            def color_risk(val):
                return {
                    "High": f"color: {RED}; font-weight: 600",
                    "Medium": f"color: {AMBER}; font-weight: 500",
                    "Low": f"color: {ACCENT}",
                }.get(val, "")

            styled_ur = disp_ur.style.map(color_risk, subset=["LEVEL"])
            st.dataframe(styled_ur, use_container_width=True, height=500)

            csv_ur = disp_ur.to_csv(index=False)
            st.download_button(
                label="Export user risk (.csv)",
                data=csv_ur, file_name="socmint_user_risk.csv",
                mime="text/csv", use_container_width=True,
            )

        with ur2:
            st.markdown('<p class="section-header">RISK DISTRIBUTION</p>', unsafe_allow_html=True)
            fig_risk = px.histogram(
                user_risk, x="risk_score", nbins=25,
                color_discrete_sequence=[RED], template="plotly_dark",
                labels={"risk_score": "Risk Score"},
            )
            fig_risk.update_layout(**PLOTLY_LAYOUT, height=250, showlegend=False)
            fig_risk.update_layout(yaxis_title="")
            st.plotly_chart(fig_risk, use_container_width=True)

            st.markdown('<p class="section-header">RISK LEVELS</p>', unsafe_allow_html=True)
            rl_counts = user_risk["risk_level"].value_counts()
            fig_rl = go.Figure(data=[go.Pie(
                labels=rl_counts.index, values=rl_counts.values, hole=0.6,
                marker_colors=[THREAT_COLORS.get(l, TEXT4) for l in rl_counts.index],
                textfont=dict(family="Geist Mono, monospace", size=10, color=TEXT2),
            )])
            fig_rl.update_layout(paper_bgcolor=BG, margin=dict(l=0, r=0, t=8, b=0), height=230,
                legend=dict(font=dict(color=TEXT2, size=10), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_rl, use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="font-family: Geist Mono, monospace; font-size: 10px; color: {TEXT4}; text-align: center; letter-spacing:0.05em;">'
    'SOCMINT Analyzer v1.0.0 · NLP + Network Analysis · Built by Kimora Taylor'
    '</p>', unsafe_allow_html=True,
)
