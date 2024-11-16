import pandas as pd

# Load the CSV file
df = pd.read_csv("trigram_generation_results_10tokens.csv")

# Extract the generation times from the third column and calculate the average
average_time = df['generation_time'].mean()

# Print the average time
print(f"The average generation time is: {average_time:.4f} seconds")



