import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict


def build_interaction_network(df):
    G = nx.Graph()

    # Add all users as nodes
    all_users = set(df["user"].unique())
    for u in all_users:
        user_posts = df[df["user"] == u]
        avg_threat = user_posts["threat_score"].mean()
        post_count = len(user_posts)
        G.add_node(u, avg_threat=avg_threat, post_count=post_count)

    # Build edges from mentions
    for _, row in df.iterrows():
        if not row["mentions"]:
            continue
        mentioned = [m.strip() for m in str(row["mentions"]).split(",") if m.strip()]
        for target in mentioned:
            if target in all_users:
                if G.has_edge(row["user"], target):
                    G[row["user"]][target]["weight"] += 1
                else:
                    G.add_edge(row["user"], target, weight=1)

    # Build edges from shared hashtag usage within tight time windows
    hashtag_users = defaultdict(list)
    for _, row in df.iterrows():
        tags = [t for t in str(row.get("hashtags", "")).split() if t.startswith("#")]
        for tag in tags:
            hashtag_users[tag].append((row["user"], row["timestamp"]))

    for tag, user_times in hashtag_users.items():
        for i in range(len(user_times)):
            for j in range(i + 1, len(user_times)):
                u1, t1 = user_times[i]
                u2, t2 = user_times[j]
                if u1 != u2 and abs((t1 - t2).total_seconds()) < 3600:  # within 1 hour
                    if G.has_edge(u1, u2):
                        G[u1][u2]["weight"] += 0.5
                    else:
                        G.add_edge(u1, u2, weight=0.5)

    return G


def detect_communities(G):
    if len(G.nodes()) == 0:
        return {}, []

    # Use Louvain-style greedy modularity
    try:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
    except Exception:
        communities = [set(G.nodes())]

    community_map = {}
    community_list = []
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
        community_list.append({
            "community_id": idx,
            "members": sorted(comm),
            "size": len(comm),
        })

    return community_map, community_list


def compute_user_risk(G, df, community_map):
    user_stats = []

    for node in G.nodes():
        user_posts = df[df["user"] == node]
        if len(user_posts) == 0:
            continue

        degree = G.degree(node, weight="weight")
        avg_threat = user_posts["threat_score"].mean()
        max_threat = user_posts["threat_score"].max()
        post_count = len(user_posts)
        flagged_count = len(user_posts[user_posts["threat_level"] != "None"])
        high_count = len(user_posts[user_posts["threat_level"] == "High"])

        # Betweenness centrality — how much this user bridges groups
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight").get(node, 0)
        except Exception:
            betweenness = 0

        # Composite risk score
        content_risk = avg_threat * 3 + (high_count / max(post_count, 1)) * 2
        network_risk = min(degree / 10.0, 1.0) + betweenness * 2
        risk_score = min((content_risk + network_risk) / 5.0, 1.0)

        if risk_score >= 0.6:
            risk_level = "High"
        elif risk_score >= 0.3:
            risk_level = "Medium"
        elif risk_score > 0.05:
            risk_level = "Low"
        else:
            risk_level = "None"

        community_id = community_map.get(node, -1)

        user_stats.append({
            "user": node,
            "post_count": post_count,
            "flagged_posts": flagged_count,
            "high_threat_posts": high_count,
            "avg_threat_score": round(avg_threat, 3),
            "max_threat_score": round(max_threat, 3),
            "network_degree": round(degree, 1),
            "betweenness": round(betweenness, 4),
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "community_id": community_id,
        })

    return pd.DataFrame(user_stats).sort_values("risk_score", ascending=False)


def get_network_layout(G):
    if len(G.nodes()) == 0:
        return {}
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42, weight="weight")
    except Exception:
        pos = nx.random_layout(G, seed=42)
    return pos


if __name__ == "__main__":
    from generate_posts import generate_posts
    from nlp_engine import analyze_posts
    import random

    np.random.seed(42)
    random.seed(42)

    df = generate_posts(1000)
    df = analyze_posts(df)

    G = build_interaction_network(df)
    community_map, communities = detect_communities(G)
    user_risk = compute_user_risk(G, df, community_map)

    print(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Communities detected: {len(communities)}")
    print(f"\nTop 10 high-risk users:")
    print(user_risk.head(10)[["user", "risk_score", "risk_level", "community_id"]].to_string())
