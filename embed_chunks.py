import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random

def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducible operations on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_nv_embed_model():
    """
    Load the NVIDIA NV-Embed-v2 model
    """
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).half()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def get_embeddings(chunks: List[str], model, device, batch_size: int = 16) -> np.ndarray:
    """
    Generate embeddings for given chunks using the NV-Embed model in batches
    """
    passage_prefix = ""
    max_length = 32768
    all_embeddings = []
    
    # Process chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch_chunks = chunks[i:i + batch_size]
        
        # Generate embeddings for current batch
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_chunks, 
                instruction=passage_prefix, 
                max_length=max_length
            )
            # Move to CPU immediately to free GPU memory
            batch_embeddings = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    return np.concatenate(all_embeddings, axis=0)

def process_chunks(transformed_plot: str) -> List[str]:
    """
    Split the transformed plot into chunks based on the <chunk> token,
    handling potential newlines before or after the token.
    """
    if pd.isna(transformed_plot) or not isinstance(transformed_plot, str):
        return []
    
    # Replace any variations of newline + <chunk> + newline with a standard delimiter
    normalized_plot = transformed_plot.replace('\n<chunk>', '<chunk>')
    normalized_plot = normalized_plot.replace('<chunk>\n', '<chunk>')
    
    chunks = normalized_plot.split('<chunk>')
    # Clean each chunk by stripping whitespace and filtering out empty chunks
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Load the transformed plots
    input_file = "movie_data_transformed_1500_plots.csv"
    df = pd.read_csv(input_file)
    
    # Load model
    print("Loading NV-Embed model...")
    model, device = load_nv_embed_model()
    
    # Initialize lists to store results
    all_movie_ids = []
    all_chunk_indices = []
    all_chunks = []
    
    print("Processing chunks...")
    # First collect all chunks
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Collecting chunks"):
        chunks = process_chunks(row['transformed_plot'])
        for chunk_idx, chunk in enumerate(chunks):
            all_movie_ids.append(row['movie_id'])
            all_chunk_indices.append(chunk_idx)
            all_chunks.append(chunk)
    
    print(f"Total chunks to process: {len(all_chunks)}")
    
    # Generate embeddings in batches
    all_embeddings = get_embeddings(all_chunks, model, device)
    
    # Create DataFrame with results
    embeddings_df = pd.DataFrame({
        'movie_id': all_movie_ids,
        'chunk_index': all_chunk_indices,
        'chunk_text': all_chunks,
        'embedding': list(all_embeddings)
    })
    
    # Save results
    output_file = "movie_chunks_embeddings.csv"
    embeddings_df.to_csv(output_file, index=False)
    
    # Save embeddings separately as numpy array for easier loading
    np.save("movie_chunks_embeddings.npy", all_embeddings)
    
    print(f"Processing complete! Results saved to {output_file}")
    print(f"Total chunks processed: {len(embeddings_df)}")
    print(f"Embeddings shape: {all_embeddings.shape}")

if __name__ == "__main__":
    main() 