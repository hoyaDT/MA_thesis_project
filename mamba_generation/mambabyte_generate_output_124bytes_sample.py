import torch
import numpy as np
import pandas as pd
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Initialize model
model = MambaLMHeadModel.from_pretrained("JunxiongWang/MambaByte_PG19_972M").to("cuda").to(torch.float32)

# Load prompts from file
file_path = "wikitext_random_prefixes.txt"
with open(file_path, "r") as file:
    prompts = [line.strip() for line in file][:101]

# List to store results
results = []

# Generate text for each prompt
for prompt in prompts:
    # Encode the prompt to bytes
    prompt_bytes = np.frombuffer(prompt.encode('utf-8'), dtype=np.uint8)
    input_ids = torch.from_numpy(prompt_bytes[None, :].copy()).long().to("cuda")

    # Record start time
    start_time = time.time()

    # Generate text with MambaByte
    sample = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids[0]) + 124,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=True,
        temperature=1,
        top_k=256,
        top_p=0.9,
    )

    # Record end time
    end_time = time.time()
    generation_time = end_time - start_time

    # Slice to get only the generated part (excluding prompt)
    generated_ids = sample.sequences[0][len(input_ids[0]):]  # Exclude the prompt length
    generated_bytes = bytes(generated_ids.tolist()).decode('utf-8', errors='ignore')

    # Append result to list
    results.append({
        "prompt": prompt,
        "generated_text": generated_bytes,
        "generation_time": generation_time
    })

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("mambabyte_generation_results_124bytes_sample.csv", index=False)

print("Generation completed. Results saved to mambabyte_generation_results_124bytes_sample.csv.")
