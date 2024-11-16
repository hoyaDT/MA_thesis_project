import kenlm
from transformers import GPT2Tokenizer
import numpy as np
import random
import time
import pandas as pd

# Initialize the GPT-2 tokenizer and KenLM model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = kenlm.LanguageModel('bigram_model.binary')

# Function to load vocabulary from the ARPA file
def get_vocabulary_from_arpa(arpa_file):
    vocabulary = set()
    in_ngrams = False

    with open(arpa_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith("\\1-grams"):
                in_ngrams = True
                continue

            if line.startswith("\\2-grams") or line.startswith("\\end\\"):
                break

            if in_ngrams and line:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[1]
                    if token.isdigit():
                        vocabulary.add(int(token))
    return list(vocabulary)

# Load vocabulary
vocabulary = get_vocabulary_from_arpa('bigram_model.arpa')

# Text generation function
def generate_text_with_ngram(prompt, model, tokenizer, vocabulary, max_len=10, top_k=1, top_p=1.0, temperature=1.0):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for _ in range(max_len):
        context = token_ids[-1:]
        context_str = " ".join(map(str, context))

        scores = []
        for token_id in vocabulary:
            token_id_str = str(token_id)
            score = model.score(context_str + " " + token_id_str, bos=False, eos=False)
            scores.append((token_id, score))

        scores = [(token_id, score / temperature) for token_id, score in scores]
        token_ids_scores = np.array([score for _, score in scores])
        
        probabilities = np.exp(token_ids_scores - np.max(token_ids_scores))
        probabilities = probabilities / probabilities.sum()

        if top_k > 0:
            sorted_indices = np.argsort(probabilities)[-top_k:]
            probabilities = probabilities[sorted_indices]
            scores = [scores[i] for i in sorted_indices]
        
        if top_p > 0.0:
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_probs = probabilities[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff_index = np.searchsorted(cumulative_probs, top_p)
            sorted_indices = sorted_indices[:cutoff_index + 1]
            probabilities = probabilities[sorted_indices]
            scores = [scores[i] for i in sorted_indices]

        probabilities = probabilities / probabilities.sum()

        next_token_index = np.random.choice(len(scores), p=probabilities)
        best_token_id = scores[next_token_index][0]

        token_ids.append(best_token_id)

    generated_text = tokenizer.decode(token_ids)
    return generated_text

# Load prompts from file
file_path = "wikitext_random_prefixes.txt"
with open(file_path, "r") as file:
    prompts = [line.strip() for line in file][1:101]  
           
# Parameters for generation
max_len = 30
top_k = 1
top_p = 1.0
temperature = 1.0

# List to store results
results = []

for prompt in prompts:
    start_time = time.time()  # Record the start time
    generated_text = generate_text_with_ngram(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        max_len=max_len,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    # Remove prompt from the generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    # Append the result
    results.append({
        "prompt": prompt,
        "generated_text": generated_text,
        "generation_time": elapsed_time
    })

# Convert results to a DataFrame for easy saving and viewing
df = pd.DataFrame(results)
df.to_csv("bigram_generation_results_30tokens.csv", index=False)  # Save results to a CSV file
print(df)  # Print results to the console
