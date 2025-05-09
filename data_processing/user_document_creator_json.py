import csv
from collections import defaultdict
import json

# Group posts by user
user_posts = defaultdict(list)
user_labels = {}

with open("data/wcpr_mypersonality.csv", mode="r", newline="", encoding="windows-1252") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        auth_id = row["#AUTHID"]
        status = row["STATUS"]
        cNEU = row["cNEU"]

        # Collect all posts under the same user
        user_posts[auth_id].append(status)

        # Save their cNEU label (assumes it's the same for all their posts)
        if auth_id not in user_labels:
            user_labels[auth_id] = cNEU

# Create final structured data
user_data = []
for user_id in user_posts:
    all_text = " ".join(user_posts[user_id])
    label = user_labels[user_id]
    user_data.append({
        "user_id": user_id,
        "text": all_text,
        "label": label
    })

# Save to JSON
with open("grouped_users.json", mode="w", encoding="utf-8") as f:
    json.dump(user_data, f, indent=4)
