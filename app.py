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
    page_title="SOCMINT Analyzer | Social Media Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e17;
    color: #c4cdd8;
  }
  .main { background: #0a0e17; }
  h1, h2, h3 { font-family: 'JetBrains Mono', monospace; color: #e2e8f0; }

  .section-header {
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  .metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-label { color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; letter-spacing: 2px; margin-bottom: 4px; }
  .metric-value { color: #f1f5f9; font-family: 'Inter', sans-serif; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
  .metric-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; margin-top: 4px; }
  .metric-delta.danger { color: #f43f5e; }
  .metric-delta.warning { color: #f59e0b; }
  .metric-delta.success { color: #22c55e; }
  .metric-delta.info { color: #38bdf8; }

  .threat-high { color: #f43f5e; font-weight: 700; }
  .threat-medium { color: #f59e0b; font-weight: 600; }
  .threat-low { color: #38bdf8; }
  .threat-none { color: #64748b; }

  .post-card {
    background: #0f172a;
    border-left: 3px solid #f43f5e;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
  }
  .post-card.medium { border-left-color: #f59e0b; }
  .post-card.low { border-left-color: #38bdf8; }

  .stSidebar { background: #060a12 !important; }
  .stButton button {
    background: linear-gradient(90deg, #0ea5e9, #0284c7);
    color: white;
    border: 1px solid #0369a1;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    font-size: 0.72rem;
  }
  .stButton button:hover { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }

  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stSelectbox"] label { color: #38bdf8; font-size: 0.72rem; letter-spacing: 2px; }

  .stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────
hc1, hc2, hc3 = st.columns([1, 5, 1])
with hc1:
    st.markdown("## 🔍")
with hc2:
    st.markdown("# SOCMINT ANALYZER")
    st.markdown('<p class="section-header">Social Media Intelligence — Threat Detection & Network Analysis</p>', unsafe_allow_html=True)
with hc3:
    st.markdown(
        '<p style="color:#22c55e; font-family: JetBrains Mono; font-size:0.72rem; margin-top:20px;">'
        '● SCANNING</p>', unsafe_allow_html=True
    )

st.markdown("---")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙ Control Panel</p>', unsafe_allow_html=True)

    n_posts = st.slider("Post Volume", 500, 5000, 2000, step=250)

    if st.button("🔄 INGEST & ANALYZE", use_container_width=True):
        st.session_state["run_seed"] = int(time.time())
        st.cache_data.clear()

    st.markdown("---")
    st.markdown('<p class="section-header">🔍 Filters</p>', unsafe_allow_html=True)

    threat_filter = st.multiselect(
        "THREAT LEVEL",
        ["High", "Medium", "Low", "None"],
        default=["High", "Medium", "Low", "None"],
    )

    view_mode = st.radio(
        "DASHBOARD VIEW",
        ["Overview", "Flagged Posts", "Network Graph", "User Risk"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-family: JetBrains Mono; font-size: 0.6rem; color: #334155;">'
        'SOCMINT Analyzer v1.0.0<br>NLP + NetworkX Engine<br>© 2026 Kimora Taylor</p>',
        unsafe_allow_html=True,
    )


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

with st.spinner("🔍 Ingesting posts and running NLP analysis..."):
    df = run_pipeline(n_posts, current_seed)

# Apply threat filter
filtered = df[df["threat_level"].isin(threat_filter)]
flagged = filtered[filtered["threat_level"] != "None"]

# ── METRICS ROW ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

total = len(filtered)
flagged_count = len(flagged)
high_count = len(filtered[filtered["threat_level"] == "High"])
med_count = len(filtered[filtered["threat_level"] == "Medium"])
unique_flagged_users = flagged["user"].nunique()

with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">POSTS ANALYZED</div>
        <div class="metric-value">{total:,}</div>
        <div class="metric-delta info">ingested feed</div>
    </div>""", unsafe_allow_html=True)
with m2:
    pct = (flagged_count / total * 100) if total else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">FLAGGED POSTS</div>
        <div class="metric-value">{flagged_count:,}</div>
        <div class="metric-delta warning">{pct:.1f}% of total</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">HIGH THREAT</div>
        <div class="metric-value">{high_count}</div>
        <div class="metric-delta danger">{"⚠ Immediate review" if high_count > 0 else "✓ Clear"}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MEDIUM THREAT</div>
        <div class="metric-value">{med_count}</div>
        <div class="metric-delta warning">monitoring</div>
    </div>""", unsafe_allow_html=True)
with m5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">FLAGGED USERS</div>
        <div class="metric-value">{unique_flagged_users}</div>
        <div class="metric-delta info">unique accounts</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VIEW: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Overview":
    ov1, ov2, ov3 = st.columns([2, 1.5, 1.5])

    with ov1:
        st.markdown('<p class="section-header">📊 Threat Activity Timeline (7 days)</p>', unsafe_allow_html=True)
        tl = filtered.copy()
        tl["day_hour"] = tl["timestamp"].dt.floor("6h")
        timeline = tl.groupby(["day_hour", "threat_level"]).size().reset_index(name="count")
        fig_tl = px.bar(
            timeline, x="day_hour", y="count", color="threat_level",
            color_discrete_map={"High": "#f43f5e", "Medium": "#f59e0b", "Low": "#38bdf8", "None": "#1e3a5f"},
            template="plotly_dark",
            labels={"day_hour": "Time", "count": "Posts", "threat_level": "Level"},
        )
        fig_tl.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
            margin=dict(l=0, r=0, t=10, b=0), bargap=0.15, height=270,
            legend=dict(font=dict(color="#c4cdd8", size=9), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9)),
            yaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9)),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    with ov2:
        st.markdown('<p class="section-header">🎯 Threat Level Breakdown</p>', unsafe_allow_html=True)
        tc = flagged["threat_level"].value_counts()
        if len(tc) > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=tc.index, values=tc.values, hole=0.6,
                marker_colors=["#f43f5e", "#f59e0b", "#38bdf8"][:len(tc)],
                textfont=dict(family="JetBrains Mono", size=9, color="white"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )])
            fig_pie.update_layout(
                paper_bgcolor="#0a0e17", margin=dict(l=0, r=0, t=10, b=0), height=270,
                legend=dict(font=dict(color="#c4cdd8", size=9), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No flagged posts with current filters.")

    with ov3:
        st.markdown('<p class="section-header">💬 Sentiment Distribution</p>', unsafe_allow_html=True)
        fig_sent = px.histogram(
            filtered, x="sentiment_polarity", nbins=40,
            color_discrete_sequence=["#38bdf8"],
            template="plotly_dark",
            labels={"sentiment_polarity": "Sentiment (← Negative | Positive →)"},
        )
        fig_sent.add_vline(x=0, line_dash="dash", line_color="#f43f5e", opacity=0.5)
        fig_sent.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
            margin=dict(l=0, r=0, t=10, b=0), height=270, showlegend=False,
            xaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9)),
            yaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9), title="Count"),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # Second row — keyword heatmap & top hashtags
    ov4, ov5 = st.columns(2)

    with ov4:
        st.markdown('<p class="section-header">🔥 Top Flagged Users</p>', unsafe_allow_html=True)
        top_users = flagged["user"].value_counts().head(12).reset_index()
        top_users.columns = ["user", "flagged_posts"]
        if len(top_users) > 0:
            fig_tu = px.bar(
                top_users, x="flagged_posts", y="user", orientation="h",
                color="flagged_posts", color_continuous_scale=["#0f172a", "#f43f5e"],
                template="plotly_dark", labels={"flagged_posts": "Flagged Posts", "user": "User"},
            )
            fig_tu.update_layout(
                paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
                margin=dict(l=0, r=0, t=10, b=0), height=320,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9)),
                yaxis=dict(tickfont=dict(family="JetBrains Mono", color="#c4cdd8", size=10)),
            )
            st.plotly_chart(fig_tu, use_container_width=True)

    with ov5:
        st.markdown('<p class="section-header"># Trending Hashtags in Flagged Posts</p>', unsafe_allow_html=True)
        all_tags = " ".join(flagged["hashtags"].dropna().astype(str)).split()
        all_tags = [t for t in all_tags if t.startswith("#")]
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(12).reset_index()
            tag_counts.columns = ["hashtag", "count"]
            fig_tags = px.bar(
                tag_counts, x="count", y="hashtag", orientation="h",
                color="count", color_continuous_scale=["#0f172a", "#f59e0b"],
                template="plotly_dark", labels={"count": "Mentions", "hashtag": "Hashtag"},
            )
            fig_tags.update_layout(
                paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
                margin=dict(l=0, r=0, t=10, b=0), height=320,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9)),
                yaxis=dict(tickfont=dict(family="JetBrains Mono", color="#c4cdd8", size=10)),
            )
            st.plotly_chart(fig_tags, use_container_width=True)
        else:
            st.info("No hashtags found in flagged posts.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: FLAGGED POSTS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Flagged Posts":
    st.markdown('<p class="section-header">🚨 Flagged Posts — Sorted by Threat Score</p>', unsafe_allow_html=True)

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
            "TIMESTAMP", "USER", "POST TEXT", "THREAT", "SCORE",
            "SENTIMENT", "HIGH KEYWORDS", "MED KEYWORDS",
            "COORD PATTERNS", "LIKES", "REPOSTS",
        ]

        def color_threat(val):
            colors = {
                "High": "color: #f43f5e; font-weight: bold",
                "Medium": "color: #f59e0b; font-weight: bold",
                "Low": "color: #38bdf8",
            }
            return colors.get(val, "")

        styled = disp.style.map(color_threat, subset=["THREAT"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = disp.to_csv(index=False)
        st.download_button(
            label="📥 EXPORT FLAGGED POSTS (CSV)",
            data=csv, file_name="socmint_flagged_posts.csv",
            mime="text/csv", use_container_width=True,
        )
    else:
        st.info("No posts match the current threat filters.")

    # Highlighted high-threat posts
    high_posts = flagged[flagged["threat_level"] == "High"].sort_values("threat_score", ascending=False).head(10)
    if len(high_posts) > 0:
        st.markdown('<p class="section-header">⚠ High-Threat Post Details</p>', unsafe_allow_html=True)
        for _, row in high_posts.iterrows():
            kw_display = row["high_keywords"]
            if row["med_keywords"]:
                kw_display += (", " if kw_display else "") + row["med_keywords"]
            st.markdown(f"""
            <div class="post-card">
                <span style="color:#f43f5e; font-weight:700;">⚠ HIGH</span>
                &nbsp;·&nbsp; <span style="color:#38bdf8;">@{row['user']}</span>
                &nbsp;·&nbsp; <span style="color:#64748b;">{row['timestamp'].strftime('%m/%d %H:%M')}</span>
                &nbsp;·&nbsp; Score: <span style="color:#f43f5e;">{row['threat_score']:.2f}</span>
                <br><br>
                <span style="color:#e2e8f0;">"{row['text']}"</span>
                <br><br>
                <span style="color:#64748b; font-size:0.75rem;">
                    Keywords: {kw_display or 'none'} &nbsp;|&nbsp;
                    Sentiment: {row['sentiment_polarity']:.2f} &nbsp;|&nbsp;
                    Coord patterns: {row['coordination_patterns']} &nbsp;|&nbsp;
                    Engagement: {row['likes']}♥ {row['reposts']}↻
                </span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Network Graph":
    st.markdown('<p class="section-header">🕸 User Interaction Network — Community Detection</p>', unsafe_allow_html=True)

    with st.spinner("Building network graph..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(G.nodes()) == 0:
        st.warning("No network data available.")
    else:
        # Build plotly network graph
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.4, color="#1e3a5f"),
            hoverinfo="none",
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

            node_text.append(
                f"@{node}<br>Risk: {risk_lvl} ({risk:.2f})<br>"
                f"Community: {comm}<br>Connections: {degree:.0f}"
            )
            node_color.append(risk)
            node_size.append(max(6, min(degree * 2 + 5, 30)))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale=[[0, "#1e3a5f"], [0.3, "#38bdf8"], [0.6, "#f59e0b"], [1, "#f43f5e"]],
                colorbar=dict(
                    title="Risk", tickfont=dict(color="#c4cdd8"),
                    titlefont=dict(color="#c4cdd8"),
                ),
                line=dict(width=0.5, color="#0a0e17"),
            ),
            text=[n if G.degree(n, weight="weight") > 3 else "" for n in G.nodes()],
            textposition="top center",
            textfont=dict(family="JetBrains Mono", size=8, color="#c4cdd8"),
            hovertext=node_text,
            hoverinfo="text",
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
            margin=dict(l=0, r=0, t=10, b=0), height=550,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Community summary
        st.markdown('<p class="section-header">👥 Detected Communities</p>', unsafe_allow_html=True)
        net1, net2 = st.columns(2)

        with net1:
            for comm in sorted(communities, key=lambda c: c["size"], reverse=True)[:8]:
                members = comm["members"]
                # Get avg risk for community
                comm_risk = user_risk[user_risk["community_id"] == comm["community_id"]]
                avg_risk = comm_risk["risk_score"].mean() if len(comm_risk) > 0 else 0
                risk_color = "#f43f5e" if avg_risk > 0.5 else "#f59e0b" if avg_risk > 0.2 else "#38bdf8"
                member_str = ", ".join([f"@{m}" for m in members[:6]])
                if len(members) > 6:
                    member_str += f" +{len(members) - 6} more"
                st.markdown(f"""
                <div style="background:#0f172a; border-left:3px solid {risk_color};
                     border-radius:6px; padding:10px 14px; margin:6px 0; font-size:0.8rem;">
                    <span style="color:{risk_color}; font-weight:700;">Community {comm['community_id']}</span>
                    &nbsp;·&nbsp; {comm['size']} members
                    &nbsp;·&nbsp; Avg risk: <span style="color:{risk_color};">{avg_risk:.2f}</span>
                    <br><span style="color:#64748b; font-size:0.72rem;">{member_str}</span>
                </div>
                """, unsafe_allow_html=True)

        with net2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NETWORK STATS</div>
                <br>
                <span style="color:#38bdf8;">Nodes:</span> {len(G.nodes())}<br>
                <span style="color:#38bdf8;">Edges:</span> {len(G.edges())}<br>
                <span style="color:#38bdf8;">Communities:</span> {len(communities)}<br>
                <span style="color:#38bdf8;">Density:</span> {nx.density(G):.4f}<br>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: USER RISK
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "User Risk":
    st.markdown('<p class="section-header">👤 User Risk Assessment</p>', unsafe_allow_html=True)

    with st.spinner("Computing user risk profiles..."):
        G, community_map, communities, user_risk, pos = run_network(n_posts, current_seed)

    if len(user_risk) == 0:
        st.warning("No user data available.")
    else:
        ur1, ur2 = st.columns([2, 1])

        with ur1:
            disp_ur = user_risk.copy()
            disp_ur.columns = [
                "USER", "POSTS", "FLAGGED", "HIGH THREAT", "AVG SCORE",
                "MAX SCORE", "NET DEGREE", "BETWEENNESS", "RISK SCORE",
                "RISK LEVEL", "COMMUNITY",
            ]

            def color_risk(val):
                return {
                    "High": "color: #f43f5e; font-weight: bold",
                    "Medium": "color: #f59e0b; font-weight: bold",
                    "Low": "color: #38bdf8",
                }.get(val, "")

            styled_ur = disp_ur.style.map(color_risk, subset=["RISK LEVEL"])
            st.dataframe(styled_ur, use_container_width=True, height=500)

            csv_ur = disp_ur.to_csv(index=False)
            st.download_button(
                label="📥 EXPORT USER RISK (CSV)",
                data=csv_ur, file_name="socmint_user_risk.csv",
                mime="text/csv", use_container_width=True,
            )

        with ur2:
            st.markdown('<p class="section-header">🎯 Risk Score Distribution</p>', unsafe_allow_html=True)
            fig_risk = px.histogram(
                user_risk, x="risk_score", nbins=25,
                color_discrete_sequence=["#f43f5e"],
                template="plotly_dark",
                labels={"risk_score": "Risk Score"},
            )
            fig_risk.update_layout(
                paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
                margin=dict(l=0, r=0, t=10, b=0), height=250,
                showlegend=False,
                xaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9)),
                yaxis=dict(gridcolor="#111827", tickfont=dict(color="#38bdf8", size=9), title="Users"),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Risk level pie
            st.markdown('<p class="section-header">📊 Risk Levels</p>', unsafe_allow_html=True)
            rl_counts = user_risk["risk_level"].value_counts()
            fig_rl = go.Figure(data=[go.Pie(
                labels=rl_counts.index, values=rl_counts.values, hole=0.55,
                marker_colors=["#f43f5e", "#f59e0b", "#38bdf8", "#1e3a5f"][:len(rl_counts)],
                textfont=dict(family="JetBrains Mono", size=9, color="white"),
            )])
            fig_rl.update_layout(
                paper_bgcolor="#0a0e17", margin=dict(l=0, r=0, t=10, b=0), height=220,
                legend=dict(font=dict(color="#c4cdd8", size=9), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_rl, use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-family: JetBrains Mono; font-size: 0.6rem; color: #334155; text-align: center;">'
    'SOCMINT Analyzer v1.0.0 — NLP + Network Analysis Engine | '
    'Python + NLTK + NetworkX + Scikit-learn + Streamlit | Built by Kimora Taylor'
    '</p>', unsafe_allow_html=True,
)
