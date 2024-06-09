import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-large-p2")

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    sentiment = "positive" if predicted_class_id == 1 else "negative"
    return sentiment
