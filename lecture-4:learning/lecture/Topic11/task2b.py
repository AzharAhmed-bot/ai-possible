from transformers import pipeline

# Load pretrained transformer pipeline for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

texts = [
    "I love this product, it works great!",
    "This is the worst experience I've ever had.",
    "It's okay, nothing special."
]

results = sentiment_analyzer(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")