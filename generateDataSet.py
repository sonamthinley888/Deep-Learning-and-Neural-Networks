import random
import pandas as pd

# Define lists of positive and negative words
positive_words = ["love", "great", "awesome", "excellent", "fantastic", "happy", "joy", "amazing", "good", "positive"]
negative_words = ["hate", "bad", "terrible", "awful", "worst", "sad", "horrible", "angry", "negative", "boring"]

# Generate synthetic sentences
def generate_sentence(sentiment="positive"):
    if sentiment == "positive":
        sentence = " ".join(random.sample(positive_words, random.randint(3, 5)))  # Randomly pick 3-5 words
        label = 1  # Positive sentiment
    else:
        sentence = " ".join(random.sample(negative_words, random.randint(3, 5)))  # Randomly pick 3-5 words
        label = 0  # Negative sentiment
    return sentence, label

# Generate a large dataset
num_samples = 1000  # You can adjust this to generate more data
texts = []
labels = []

for _ in range(num_samples):
    sentiment = random.choice(["positive", "negative"])  # Randomly choose sentiment
    sentence, label = generate_sentence(sentiment)
    texts.append(sentence)
    labels.append(label)

# Create a DataFrame to store the dataset
data = pd.DataFrame({
    'text': texts,
    'label': labels
})

# Save the dataset to a CSV file
data.to_csv('synthetic_sentiment_dataset.csv', index=False)

print("Dataset exported to 'synthetic_sentiment_dataset.csv'")
