import spacy
import json
# Load spaCy English tokenizer/lemmatizer
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)


# Load grouped user data
with open("data/grouped_users.json", "r", encoding="utf-8") as f:
    user_data = json.load(f)

# Preprocess all texts
for user in user_data:
    user["clean_text"] = preprocess_text(user["text"])

# Save new version
with open("grouped_users_clean.json", "w", encoding="utf-8") as f:
    json.dump(user_data, f, indent=4)
