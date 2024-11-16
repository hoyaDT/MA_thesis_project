import pexpect
import re
import pandas as pd
import time

# Load prompts from file
file_path = "wikitext_random_prefixes.txt"
with open(file_path, "r") as file:
    prompts = [line.strip() for line in file][:20]  # Read first 20 lines

# Define a parsing function to extract the top candidate text and its probability
def get_top_candidate(output):
    # Use regex to retrieve the candidate word and its probability
    matches = re.findall(r"p=([\d.]+) \(\d+/\d+\), k=\d+: (.+)", output)
    if matches:
        probability = float(matches[0][0])  
        candidate_text = matches[0][1].strip()  
        return candidate_text, probability
    return None, None

# Parameters
num_words_to_generate = 10

# List to store results
results = []

# Generate text for each prompt
for prompt in prompts:
    current_prompt = prompt  # Start with the initial prompt
    start_time = time.time()  # Record the start time

    for _ in range(num_words_to_generate):
        # Restart the infinigram process for each query
        process = pexpect.spawn(
            "./infinigram --train_file extracted_text_2_5B_chars.txt --out_dir output --tokenizer_config tokenizer.json",
            encoding='utf-8'
        )

        process.expect("enter query:")  # Wait for infinigram's input prompt
        process.sendline(current_prompt)  # Enter the current prompt
        process.expect("enter query:")  # Capture the generated candidate word and its probability distribution
        output = process.before  # Capture output

        # Get the top candidate text and update the prompt directly
        top_candidate, prob = get_top_candidate(output)
        if top_candidate:
            current_prompt += " " + top_candidate  # Append the candidate text to the prompt
        else:
            process.close()
            break  # Exit the loop early if no candidate is found

        process.close()

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time

    # Remove initial prompt from the generated text
    generated_text = current_prompt[len(prompt):].strip()

    # Append the result
    results.append({
        "prompt": prompt,
        "generated_text": generated_text,
        "generation_time": elapsed_time
    })

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("infinigram_generation_results_10tokens.csv", index=False)
print("Generation completed. Results saved to infinigram_generation_results_10tokens.csv.")
print(df)  # Print results to the console






