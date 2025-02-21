import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1)
    
    sentiment_score = torch.argmax(probabilities).item()
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    
    return labels[sentiment_score], round(probabilities.max().item(), 4)
