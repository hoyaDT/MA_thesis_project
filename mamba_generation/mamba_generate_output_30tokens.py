import torch
import time
import pandas as pd
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

# Load prompts from file
file_path = "wikitext_random_prefixes.txt"
with open(file_path, "r") as file:
    prompts = [line.strip() for line in file][:101] 

# Set maximum number of new tokens for generation
max_new_tokens = 30

# List to store results
results = []

# Generate text for each prompt and measure time
for prompt in prompts:
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Record start time
    start_time = time.time()

    # Generate text using greedy decoding
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Record end time
    end_time = time.time()
    generation_time = end_time - start_time

    # Remove the input prompt part from generated output
    new_token_ids = output_ids[0, input_ids.shape[1]:]  # Slice to keep only new tokens
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Append result to list
    results.append({
        "prompt": prompt,
        "generated_text": generated_text,
        "generation_time": generation_time
    })

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("mamba_generation_results_30tokens.csv", index=False)

print("Generation completed. Results saved to mamba_generation_results_30tokens.csv.")



