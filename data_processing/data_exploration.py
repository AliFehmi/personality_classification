import json

with open("../data/grouped_users_clean.json", "r", encoding="utf-8") as f:
    user_data = json.load(f)

num_users = len(user_data)
print(f"Total number of users: {num_users}")
