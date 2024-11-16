from multiprocessing import Pool
from transformers import GPT2Tokenizer
import sys
import time

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_chunk(chunk_text):
    # Tokenize a chunk of text and return token IDs
    return tokenizer.encode(chunk_text)

def tokenize_file_in_parallel(input_file, output_file, num_processes=8, chunk_size=50*1024*1024):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        chunk_texts = []
        
        # Read the file in chunks and store the text chunks
        while True:
            text = infile.read(chunk_size)
            if not text:
                break
            chunk_texts.append(text)
        
        # Use multiprocessing to tokenize the chunks in parallel
        with Pool(processes=num_processes) as pool:
            tokenized_chunks = pool.map(tokenize_chunk, chunk_texts)

        # Write tokenized chunks to the output file
        for token_ids in tokenized_chunks:
            outfile.write(" ".join(map(str, token_ids)) + " ")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Measure start time
    start_time = time.time()
    
    # Tokenize file in parallel
    tokenize_file_in_parallel(input_file, output_file)
    
    # Measure end time
    end_time = time.time()
    
    total_time = end_time - start_time

    print(f"Tokenized text saved to {output_file}")
    print(f"Total time taken: {total_time:.2f} seconds") #Total time taken: 485.73 seconds


