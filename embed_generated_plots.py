import pandas as pd
from openai import OpenAI
import time
from typing import Optional
import argparse
import torch
from transformers import AutoModel
import numpy as np
from tqdm import tqdm
import backoff
from openai import RateLimitError
import random

@backoff.on_exception(backoff.expo, RateLimitError)
def chat_completion_with_backoff(client, messages, **kwargs):
    """
    Wrapper function for chat completions with exponential backoff retry logic
    """
    return client.chat.completions.create(
        messages=messages,
        **kwargs
    )

def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def chunk_and_anonymize_plot(client, plot: Optional[str]) -> str:
    """
    Calls OpenAI API to anonymize proper nouns and chunk the plot.
    Returns empty string if plot is None or empty.
    """
    if pd.isna(plot) or not str(plot).strip():
        return ""
    
    try:
        prompt = """Take the following plot and anonymize all proper nouns by replacing them with a single uppercase placeholder letter such as A, B, X, Y or Z, ensuring consistent substitution across the entire text. Divide the plot into logically consistent sections (chunks) based on context or continuity, and mark each section by inserting the token <chunk>. Return only the processed text with placeholders and chunk markers. Do not include explanations, examples, or additional text in the output.

Plot: """ + str(plot)

        response = chat_completion_with_backoff(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.2,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error processing plot: {e}")
        return ""

def load_nv_embed_model():
    """
    Load the NVIDIA NV-Embed-v2 model
    """
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def get_embeddings(chunks: list[str], model, device, batch_size: int = 16) -> np.ndarray:
    """
    Generate embeddings for given chunks using the NV-Embed model
    """
    passage_prefix = ""
    max_length = 32768
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch_chunks = chunks[i:i + batch_size]
        
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_chunks, 
                instruction=passage_prefix, 
                max_length=max_length
            )
            batch_embeddings = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.concatenate(all_embeddings, axis=0)

def process_chunks(transformed_plot: str) -> list[str]:
    """
    Split the transformed plot into chunks based on the <chunk> token
    """
    if pd.isna(transformed_plot) or not isinstance(transformed_plot, str):
        return []
    
    normalized_plot = transformed_plot.replace('\n<chunk>', '<chunk>')
    normalized_plot = normalized_plot.replace('<chunk>\n', '<chunk>')
    
    chunks = normalized_plot.split('<chunk>')
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def main():
    parser = argparse.ArgumentParser(description='Process and embed generated plots')
    parser.add_argument('--input_file', type=str, default='generated_plots_10.csv',
                      help='Input file containing generated plots')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    set_seed(args.seed)
    client = OpenAI()
    
    # Read the generated plots
    df = pd.read_csv(args.input_file)
    
    # Transform plots
    print("Transforming plots...")
    df['transformed_plot'] = ''
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing plots"):
        transformed_plot = chunk_and_anonymize_plot(client, row['plot'])
        df.at[idx, 'transformed_plot'] = transformed_plot
    
    # Save transformed plots
    base_filename = args.input_file.replace('.csv', '')
    transformed_file = f"{base_filename}_transformed.csv"
    df.to_csv(transformed_file, index=False)
    
    # Load NV-Embed model
    print("\nLoading NV-Embed model...")
    model, device = load_nv_embed_model()
    
    # Process chunks and generate embeddings
    all_plot_ids = []
    all_chunk_indices = []
    all_genres = []
    all_chunks = []
    
    print("Processing chunks...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Collecting chunks"):
        chunks = process_chunks(row['transformed_plot'])
        for chunk_idx, chunk in enumerate(chunks):
            all_plot_ids.append(row['plot_id'])
            all_chunk_indices.append(chunk_idx)
            all_genres.append(row['genre'])
            all_chunks.append(chunk)
    
    print(f"Total chunks to process: {len(all_chunks)}")
    
    # Generate embeddings
    all_embeddings = get_embeddings(all_chunks, model, device)
    
    # Create DataFrame with results
    embeddings_df = pd.DataFrame({
        'plot_id': all_plot_ids,
        'chunk_index': all_chunk_indices,
        'genre': all_genres,
        'chunk_text': all_chunks,
        'embedding': list(all_embeddings)
    })
    
    # Save results
    output_file = f"{base_filename}_embeddings.csv"
    embeddings_df.to_csv(output_file, index=False)
    np.save(f"{base_filename}_embeddings.npy", all_embeddings)
    
    print(f"Processing complete! Results saved to {output_file}")
    print(f"Total chunks processed: {len(embeddings_df)}")
    print(f"Embeddings shape: {all_embeddings.shape}")

if __name__ == "__main__":
    main() 