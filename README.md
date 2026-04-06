# 🔍 SOCMINT Analyzer — Social Media Intelligence Platform

A real-world-inspired intelligence tool that analyzes social media posts to detect threats, extremist language, and suspicious coordination patterns using NLP and network analysis.

**Built by Kimora Taylor**

## What It Does

1. **Ingests Social Media Data** — Generates realistic simulated Twitter/X-style posts including benign, suspicious, extremist, and coordinated activity
2. **NLP Threat Analysis** — Keyword detection (threat lexicons), sentiment analysis (aggressive tone detection), and coordination pattern recognition
3. **Flags Suspicious Content** — Assigns threat levels (High / Medium / Low) based on composite scoring
4. **Network Intelligence** — Detects user interaction networks, identifies communities, and maps coordination clusters using graph analysis
5. **Interactive Dashboard** — Four views: Overview, Flagged Posts, Network Graph, and User Risk Assessment

## Threat Detection Features

| Capability | Method |
|-----------|--------|
| **Threat Keywords** | Multi-tier lexicon (high/medium/low severity) |
| **Sentiment Analysis** | TextBlob polarity + subjectivity scoring |
| **Coordination Detection** | Regex patterns for operational language |
| **Aggression Scoring** | Combined negative sentiment + high subjectivity |
| **Community Detection** | Greedy modularity clustering (NetworkX) |
| **User Risk Profiling** | Composite of content risk + network centrality |

## Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data processing
- **NLTK / TextBlob** — Natural Language Processing
- **NetworkX** — Graph analysis & community detection
- **Scikit-learn** — ML utilities
- **Plotly** — Interactive visualizations
- **Streamlit** — Web dashboard

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Dashboard Views

- **Overview** — Threat timeline, sentiment distribution, top flagged users, trending hashtags
- **Flagged Posts** — Sortable table of all flagged content with threat scores, keywords, and detailed high-threat post cards
- **Network Graph** — Interactive force-directed graph showing user connections, community clusters, and risk coloring
- **User Risk** — Complete risk assessment table with network metrics, betweenness centrality, and exportable CSV

## Project Structure

```
socmint-analyzer/
├── app.py                 # Streamlit dashboard (main entry point)
├── generate_posts.py      # Synthetic social media data generator
├── nlp_engine.py          # NLP analysis (sentiment, keywords, coordination)
├── network_analysis.py    # Graph building, community detection, risk scoring
├── requirements.txt       # Python dependencies
└── README.md
```
