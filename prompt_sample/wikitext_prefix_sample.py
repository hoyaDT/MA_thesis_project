from datasets import load_dataset
from transformers import GPT2Tokenizer
import random

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the wikitext dataset (test split)
dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True)

# Concatenate all non-empty text entries
all_text = ""
for example in dataset:
    text = example['text'].strip()
    if text:
        all_text += " " + text

# Tokenize the entire concatenated text using GPT-2 tokenizer
tokens = tokenizer.tokenize(all_text)

# Collect prefixes of 6 subwords
prefixes = []
num_prefixes = 101  # Target number of prefixes
while len(prefixes) < num_prefixes and len(tokens) >= 6:
    # Randomly select a starting point for a 6-subword prefix
    start_idx = random.randint(0, len(tokens) - 6)
    prefix_tokens = tokens[start_idx:start_idx + 6]
    
    # Convert the token list back to a text prefix
    prefix = tokenizer.convert_tokens_to_string(prefix_tokens).strip()
    
    # Ensure the prefix is non-empty before adding it to the list
    if prefix:
        prefixes.append(prefix)

# Save all prefixes to a txt file
with open("wikitext_random_prefixes.txt", "w", encoding="utf-8") as f:
    for prefix in prefixes:
        f.write(prefix + "\n")

print(f"Randomly selected 6-subword prefixes have been saved to wikitext_random_prefixes.txt. Total prefixes collected: {len(prefixes)}")




