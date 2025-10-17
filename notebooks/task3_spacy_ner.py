# task3_spacy_ner.py

import spacy
import pandas as pd

# Load spaCy English model (small one)
nlp = spacy.load("en_core_web_sm")

# === STEP 1: SAMPLE TEXT DATA ===
reviews = [
    "I absolutely love my new Samsung Galaxy S24!",
    "The Apple MacBook Air is overpriced but works smoothly.",
    "Nike shoes are comfortable but delivery was slow.",
    "I bought a Sony headphone and the sound quality is perfect!",
    "Terrible experience with this Dell laptop, it crashed twice."
]

df = pd.DataFrame({"review": reviews})

# === STEP 2: PERFORM NAMED ENTITY RECOGNITION ===
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df["entities"] = df["review"].apply(extract_entities)

# === STEP 3: SIMPLE RULE-BASED SENTIMENT ANALYSIS ===
def rule_based_sentiment(text):
    text = text.lower()
    positive_words = ["love", "perfect", "great", "smooth", "comfortable"]
    negative_words = ["terrible", "slow", "crashed", "bad", "overpriced"]
    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review"].apply(rule_based_sentiment)

# === STEP 4: SHOW RESULTS ===
print("\n✅ Named Entities & Sentiment Analysis Results:\n")
print(df[["review", "entities", "sentiment"]])
