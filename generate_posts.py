import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

NORMAL_USERS = [f"user_{i:04d}" for i in range(1, 80)]
SUSPICIOUS_USERS = [f"shadow_{i:02d}" for i in range(1, 12)]
COORDINATED_GROUP_A = ["alpha_01", "alpha_02", "alpha_03", "alpha_04", "alpha_05"]
COORDINATED_GROUP_B = ["omega_01", "omega_02", "omega_03", "omega_04"]

BENIGN_POSTS = [
    "Just had the best coffee this morning ☕",
    "Anyone watching the game tonight?",
    "Beautiful sunset from my balcony today",
    "Working from home has its perks 🏠",
    "Can't believe it's already April",
    "New recipe turned out amazing! Sharing soon",
    "Happy birthday to my best friend! 🎂",
    "Morning run done ✅ feeling great",
    "This new album is incredible, highly recommend",
    "Grateful for the little things today",
    "Traffic was terrible this morning, took forever",
    "Movie night with the family 🍿",
    "Just finished a great book, need recommendations",
    "Loving the weather today, perfect for a walk",
    "Finally got my garden planted for the season",
    "Who else is excited for the weekend?",
    "Great meeting with the team today, big things coming",
    "Trying out a new restaurant downtown tonight",
    "My dog learned a new trick today 🐕",
    "Sunday brunch is the best tradition",
    "Just signed up for a 5K run next month",
    "The kids had so much fun at the park",
    "New podcast episode dropped, it's a good one",
    "Home improvement project finally done!",
    "Coffee and a good book, perfect Saturday",
    "Congrats to the graduating class! 🎓",
    "Found the cutest little shop in the neighborhood",
    "Pizza Friday is the best day of the week",
    "Can someone recommend a good plumber?",
    "So proud of my team's performance this quarter",
]

SUSPICIOUS_POSTS = [
    "They won't see it coming. Trust the plan.",
    "Meet at the spot. 3pm. You know where.",
    "The target has been identified. Waiting for the signal.",
    "Everything is in position. Ready when you are.",
    "We need to act before they find out",
    "The package will be delivered on schedule",
    "Phase two begins at dawn. No more waiting.",
    "They think they're safe. They're wrong.",
    "Don't use your phone. Switch to the backup channel.",
    "Bring everything we discussed. No exceptions.",
    "The building is unguarded after midnight",
    "We have eyes on the location. Standing by.",
    "Delete all messages after reading this",
    "Time to show them what we're capable of",
    "The operation moves forward regardless of the news",
    "All units confirm positions by midnight",
    "Keep your head down until I give the word",
    "The window of opportunity closes tomorrow",
    "Make sure nobody follows you to the meeting point",
    "Acquire the materials on the list. Be discreet.",
]

EXTREMIST_POSTS = [
    "The system is corrupt beyond repair. Only force can fix it.",
    "Rise up and fight! The time for talk is over!",
    "They want to silence us. We will not be silenced!",
    "Our enemies will pay for what they've done to our people",
    "The revolution starts now. Who's with me?",
    "Burn it all down and rebuild from the ashes",
    "Violence is the only language they understand",
    "Join the cause or you're part of the problem",
    "We are the resistance. We will prevail by any means.",
    "Death to the oppressors! Freedom for the people!",
    "Arm yourselves. The day of reckoning is near.",
    "Destroy the symbols of their power",
    "No mercy for traitors. You know who you are.",
    "The blood of patriots waters the tree. Act now.",
    "Recruitment is open. DM for details. Loyalty required.",
]

COORDINATION_PHRASES = [
    "Confirmed. Moving to phase {}.",
    "Copy that. Position {} secured.",
    "Acknowledged. ETA {} minutes.",
    "Roger. Package {} en route.",
    "Standing by at checkpoint {}.",
    "All clear on sector {}.",
]

HASHTAGS_BENIGN = ["#goodvibes", "#blessed", "#dailylife", "#fitness", "#food",
                   "#sunset", "#weekend", "#family", "#motivation", "#love"]
HASHTAGS_SUSPICIOUS = ["#theplan", "#wakeup", "#silenced", "#resist", "#truthwillout",
                       "#liberation", "#nocompromise", "#uprising", "#revolution", "#united"]


def generate_posts(n=2000):
    records = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(n):
        r = random.random()
        timestamp = base_time + timedelta(seconds=random.randint(0, 7 * 86400))

        if r < 0.60:
            # Normal post
            user = random.choice(NORMAL_USERS)
            text = random.choice(BENIGN_POSTS)
            if random.random() < 0.3:
                text += " " + random.choice(HASHTAGS_BENIGN)
            mentions = []
            if random.random() < 0.15:
                mentions = [random.choice(NORMAL_USERS)]
            label = "benign"

        elif r < 0.78:
            # Suspicious post
            user = random.choice(SUSPICIOUS_USERS)
            text = random.choice(SUSPICIOUS_POSTS)
            if random.random() < 0.4:
                text += " " + random.choice(HASHTAGS_SUSPICIOUS)
            mentions = []
            if random.random() < 0.3:
                mentions = [random.choice(SUSPICIOUS_USERS)]
            label = "suspicious"

        elif r < 0.88:
            # Extremist post
            user = random.choice(SUSPICIOUS_USERS + COORDINATED_GROUP_A[:2])
            text = random.choice(EXTREMIST_POSTS)
            if random.random() < 0.5:
                text += " " + random.choice(HASHTAGS_SUSPICIOUS)
            mentions = []
            if random.random() < 0.25:
                mentions = [random.choice(SUSPICIOUS_USERS)]
            label = "extremist"

        elif r < 0.95:
            # Coordinated group A
            user = random.choice(COORDINATED_GROUP_A)
            phase = random.randint(1, 5)
            text = random.choice(COORDINATION_PHRASES).format(phase)
            mentions = [u for u in COORDINATED_GROUP_A if u != user][:random.randint(1, 3)]
            if random.random() < 0.6:
                text += " " + random.choice(HASHTAGS_SUSPICIOUS)
            # Cluster timestamps for coordination
            cluster_base = base_time + timedelta(days=random.choice([1, 3, 5]),
                                                  hours=random.choice([2, 3, 14, 15]))
            timestamp = cluster_base + timedelta(minutes=random.randint(0, 30))
            label = "coordinated"

        else:
            # Coordinated group B
            user = random.choice(COORDINATED_GROUP_B)
            phase = random.randint(1, 5)
            text = random.choice(COORDINATION_PHRASES).format(phase)
            mentions = [u for u in COORDINATED_GROUP_B if u != user][:random.randint(1, 2)]
            if random.random() < 0.6:
                text += " " + random.choice(HASHTAGS_SUSPICIOUS)
            cluster_base = base_time + timedelta(days=random.choice([2, 4, 6]),
                                                  hours=random.choice([1, 2, 22, 23]))
            timestamp = cluster_base + timedelta(minutes=random.randint(0, 20))
            label = "coordinated"

        # Simulate engagement
        if label == "benign":
            likes = random.randint(0, 200)
            reposts = random.randint(0, 20)
        elif label == "extremist":
            likes = random.randint(50, 500)
            reposts = random.randint(10, 100)
        else:
            likes = random.randint(5, 150)
            reposts = random.randint(1, 40)

        records.append({
            "post_id": f"post_{i:06d}",
            "timestamp": timestamp,
            "user": user,
            "text": text,
            "mentions": ",".join(mentions) if mentions else "",
            "hashtags": " ".join([t for t in text.split() if t.startswith("#")]),
            "likes": likes,
            "reposts": reposts,
            "true_label": label,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    df = generate_posts(2000)
    df.to_csv("social_posts.csv", index=False)
    print(f"Generated {len(df)} posts")
    print(f"Label distribution:\n{df['true_label'].value_counts()}")
