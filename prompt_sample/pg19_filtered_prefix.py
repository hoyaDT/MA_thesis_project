from datasets import load_dataset
from transformers import GPT2Tokenizer
import re

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the dataset with streaming enabled
dataset = load_dataset("pg19", split='train', streaming=True)

# Store prefix phrases
prefixes = []
target_samples = 20  # Target is 20 prefixes
skip_sentences = 20  # Skip the first 20 sentences of each text

# Define regex to filter out text with excessive symbols or numbers
invalid_pattern = re.compile(r'[^\w\s]|[0-9]')

# Iterate through the dataset
for example in dataset:
    text = example['text']
    
    # Split into sentences
    sentences = text.split('. ')
    
    # Check if there are enough sentences to skip
    if len(sentences) > skip_sentences:
        # Skip the first 20 sentences
        remaining_sentences = sentences[skip_sentences:]
        
        # Iterate over the remaining sentences to find a valid prefix
        for sentence in remaining_sentences:
            # Clean up the sentence by removing line breaks or extra spaces
            sentence = sentence.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Tokenize the sentence using GPT-2 tokenizer
            tokens = tokenizer.tokenize(sentence)
            
            # Check if the sentence meets all criteria
            if len(tokens) >= 6 and not invalid_pattern.search(sentence) and not sentence.isupper():
                # Extract the first 6 subwords and join them into a string
                prefix = tokenizer.convert_tokens_to_string(tokens[:6]).strip()
                
                # Add the prefix to the list
                prefixes.append(prefix)
                break  # Stop after finding the first valid prefix for this text
    
    # Stop after reaching the target number of samples
    if len(prefixes) >= target_samples:
        break

# Save all prefixes to a txt file
with open("pg19_filtered_prefixes.txt", "w", encoding="utf-8") as f:
    for prefix in prefixes:
        f.write(prefix + "\n")

print("Prefixes of 6 subwords that meet the criteria have been saved to pg19_filtered_prefixes.txt")

           
           









