from datasets import load_dataset

# Load the dataset in a streamed way
dataset = load_dataset("monology/pile-uncopyrighted", split='train', streaming=True)

# Shuffle the dataset with a buffer
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

# Initialize variables
char_count = 0
max_chars = 2500000000  # 2.5B characters
text_chunks = []

# Iterate through the shuffled dataset and concatenate until reaching 2.5B characters
for example in shuffled_dataset:
    text = example['text']  # Get content from the 'text' field
    if char_count + len(text) > max_chars:
        text_chunks.append(text[:max_chars - char_count])  # Append the remaining text to hit 2.5B characters
        break
    else:
        text_chunks.append(text)
        char_count += len(text)

# Combine all text chunks into a single string
extracted_text = ''.join(text_chunks)

# Save the extracted text to a local file
with open("extracted_text_2_5B_chars.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("Extracted text saved to extracted_text_2_5B_chars.txt")
