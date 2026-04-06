import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
import nltk

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "stopwords"]:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ── Threat lexicons ──────────────────────────────────────────────────────────
THREAT_KEYWORDS = {
    "high": [
        "bomb", "attack", "kill", "destroy", "weapon", "explode", "detonate",
        "assassinate", "execute", "annihilate", "slaughter", "massacre",
        "death", "die", "armed", "shoot", "gun", "rifle", "ammunition",
        "recruit", "radicalize", "martyr", "jihad", "infidel",
    ],
    "medium": [
        "target", "strike", "operation", "phase", "mission", "position",
        "signal", "checkpoint", "sector", "package", "deliver", "acquire",
        "materials", "discreet", "surveillance", "recon", "infiltrate",
        "overthrow", "revolt", "uprising", "rebellion", "resistance",
        "mobilize", "coordinate", "rendezvous", "extraction",
    ],
    "low": [
        "suspicious", "watching", "follow", "track", "monitor", "plan",
        "secret", "covert", "underground", "hideout", "safehouse",
        "encrypt", "burner", "delete", "backup channel", "sweep",
        "unguarded", "midnight", "dawn", "silent", "shadow",
    ],
}

COORDINATION_INDICATORS = [
    r"phase\s+\d+",
    r"position\s+\d+",
    r"checkpoint\s+\d+",
    r"sector\s+\d+",
    r"eta\s+\d+",
    r"package\s+\d+",
    r"confirmed\.?\s",
    r"copy\s+that",
    r"roger\.?\s",
    r"acknowledged",
    r"standing\s+by",
    r"all\s+clear",
    r"moving\s+to",
    r"en\s+route",
]


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def compute_threat_keywords(text):
    text_lower = text.lower()
    tokens = set(re.findall(r"\b[a-z]+\b", text_lower))

    high_matches = [kw for kw in THREAT_KEYWORDS["high"] if kw in text_lower]
    med_matches = [kw for kw in THREAT_KEYWORDS["medium"] if kw in text_lower]
    low_matches = [kw for kw in THREAT_KEYWORDS["low"] if kw in text_lower]

    return high_matches, med_matches, low_matches


def detect_coordination_language(text):
    text_lower = text.lower()
    matches = []
    for pattern in COORDINATION_INDICATORS:
        if re.search(pattern, text_lower):
            matches.append(pattern)
    return matches


def extract_keywords(text, top_n=5):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]


def analyze_posts(df):
    results = []

    for _, row in df.iterrows():
        text = row["text"]

        # Sentiment
        polarity, subjectivity = analyze_sentiment(text)

        # Threat keywords
        high_kw, med_kw, low_kw = compute_threat_keywords(text)
        keyword_score = len(high_kw) * 3 + len(med_kw) * 2 + len(low_kw) * 1

        # Coordination language
        coord_matches = detect_coordination_language(text)
        coord_score = len(coord_matches) * 2

        # Aggressive tone bonus: very negative sentiment with high subjectivity
        aggression_score = 0
        if polarity < -0.3 and subjectivity > 0.5:
            aggression_score = 2
        elif polarity < -0.1:
            aggression_score = 1

        # Composite threat score (0-10 scale)
        raw_score = keyword_score + coord_score + aggression_score
        threat_score = min(raw_score / 10.0, 1.0)

        # Threat level
        if threat_score >= 0.6 or len(high_kw) >= 2:
            threat_level = "High"
        elif threat_score >= 0.3 or len(high_kw) >= 1 or len(med_kw) >= 2:
            threat_level = "Medium"
        elif threat_score > 0.05 or len(low_kw) >= 1 or len(coord_matches) >= 1:
            threat_level = "Low"
        else:
            threat_level = "None"

        results.append({
            "post_id": row["post_id"],
            "sentiment_polarity": round(polarity, 3),
            "sentiment_subjectivity": round(subjectivity, 3),
            "high_keywords": ", ".join(high_kw) if high_kw else "",
            "med_keywords": ", ".join(med_kw) if med_kw else "",
            "low_keywords": ", ".join(low_kw) if low_kw else "",
            "keyword_score": keyword_score,
            "coordination_patterns": len(coord_matches),
            "aggression_score": aggression_score,
            "threat_score": round(threat_score, 3),
            "threat_level": threat_level,
        })

    results_df = pd.DataFrame(results)
    df_merged = df.merge(results_df, on="post_id", how="left")
    return df_merged


if __name__ == "__main__":
    df = pd.read_csv("social_posts.csv", parse_dates=["timestamp"])
    df = analyze_posts(df)
    print(f"Threat level distribution:\n{df['threat_level'].value_counts()}")
    print(f"\nHigh-threat posts:")
    for _, r in df[df["threat_level"] == "High"].head(5).iterrows():
        print(f"  [{r['user']}] {r['text'][:80]}...")
