from datasets import load_dataset
from collections import Counter
from nltk.util import ngrams
import nltk
import re

# Download the NLTK tokenizer
nltk.download("punkt")

# Load the dataset and shuffle it
dataset = load_dataset("pg19", split='train', streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=1000)

# Initialize variables
text_chunks = []
sample_size = 100  # Select 100 text samples

# Randomly select 100 text samples
for i, example in enumerate(shuffled_dataset):
    text_chunks.append(example['text'])
    if i >= sample_size - 1:
        break

# Combine all text samples into one string
combined_text = ' '.join(text_chunks)

# Text preprocessing function
def preprocess_text(text):
    # Remove non-alphabet characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = nltk.word_tokenize(text)
    return tokens

# Tokenize and preprocess the text
tokens = preprocess_text(combined_text)

# Extract n-grams (phrases)
n = 3  # Set n-gram to 3, representing three-word phrases
phrases = list(ngrams(tokens, n))

# Count the 20 most common phrases
phrase_counts = Counter(phrases)
most_common_phrases = phrase_counts.most_common(20)

# Convert phrases to string format
most_common_phrases = [" ".join(phrase) for phrase, _ in most_common_phrases]

# Output the most common phrases
print("Most common phrases:", most_common_phrases)

