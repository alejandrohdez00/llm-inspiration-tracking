import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import torch.nn.functional as F
import torch
from tqdm import tqdm

def load_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load the embeddings and metadata for both generated and original plots
    """
    # Load embeddings
    generated_embeddings = np.load("generated_plots_5_embeddings.npy")
    movie_embeddings = np.load("movie_chunks_embeddings.npy")
    
    # Load metadata - make sure to load the chunk-level data
    generated_df = pd.read_csv("generated_plots_5_embeddings.csv")
    movie_chunks_df = pd.read_csv("movie_chunks_embeddings.csv")
    
    # Verify the lengths match
    assert len(movie_embeddings) == len(movie_chunks_df), "Mismatch between embeddings and metadata length"
    assert len(generated_embeddings) == len(generated_df), "Mismatch between generated embeddings and metadata length"
    
    return generated_embeddings, movie_embeddings, generated_df, movie_chunks_df

def compute_similarities(query_embeddings: np.ndarray, 
                        reference_embeddings: np.ndarray,
                        top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between query and reference embeddings
    Returns top_k similarities and indices
    """
    # Convert to torch tensors for efficient computation
    query_tensor = torch.from_numpy(query_embeddings).float()
    ref_tensor = torch.from_numpy(reference_embeddings).float()
    
    # Normalize embeddings
    query_normalized = F.normalize(query_tensor, p=2, dim=1)
    ref_normalized = F.normalize(ref_tensor, p=2, dim=1)
    
    # Compute similarities in batches to avoid memory issues
    batch_size = 1000
    all_similarities = []
    
    for i in tqdm(range(0, len(query_normalized), batch_size), desc="Computing similarities"):
        batch = query_normalized[i:i+batch_size]
        # Compute cosine similarity
        sim = torch.mm(batch, ref_normalized.t())
        all_similarities.append(sim)
    
    similarities = torch.cat(all_similarities, dim=0)
    
    # Get top-k similarities and indices
    top_k_similarities, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
    
    return top_k_similarities.numpy(), top_k_indices.numpy()

def visualize_embeddings(generated_embeddings: np.ndarray, 
                        movie_embeddings: np.ndarray,
                        generated_df: pd.DataFrame,
                        movie_df: pd.DataFrame):
    """
    Create PCA visualization of the embedding space
    """
    # Combine embeddings
    all_embeddings = np.vstack([generated_embeddings, movie_embeddings])
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(all_embeddings)
    
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)
    
    # Split back into generated and original
    n_generated = len(generated_embeddings)
    generated_reduced = reduced_embeddings[:n_generated]
    movie_reduced = reduced_embeddings[n_generated:]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot original movie chunks
    plt.scatter(movie_reduced[:, 0], movie_reduced[:, 1], 
               alpha=0.5, label='Original Movies', s=10)
    
    # Plot generated chunks with different color for each genre
    for genre in generated_df['genre'].unique():
        mask = generated_df['genre'] == genre
        genre_embeddings = generated_reduced[mask]
        plt.scatter(genre_embeddings[:, 0], genre_embeddings[:, 1], 
                   alpha=0.7, label=f'Generated ({genre})', s=30)
    
    plt.title('PCA Visualization of Plot Chunk Embeddings')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('embedding_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    print("Loading embeddings and metadata...")
    generated_embeddings, movie_embeddings, generated_df, movie_df = load_data()
    
    # Compute similarities
    print("\nComputing similarities...")
    top_k_similarities, top_k_indices = compute_similarities(generated_embeddings, movie_embeddings)
    
    # Create results DataFrame
    results = []
    for i in range(len(generated_df)):
        generated_info = {
            'plot_id': generated_df.iloc[i]['plot_id'],
            'genre': generated_df.iloc[i]['genre'],
            'chunk_index': generated_df.iloc[i]['chunk_index'],
            'generated_text': generated_df.iloc[i]['chunk_text']
        }
        
        # Add top 5 matches
        for k in range(5):
            movie_idx = top_k_indices[i, k]
            similarity = top_k_similarities[i, k]
            
            match_info = {
                f'match_{k+1}_movie_id': movie_df.iloc[movie_idx]['movie_id'],
                f'match_{k+1}_chunk_index': movie_df.iloc[movie_idx]['chunk_index'],
                f'match_{k+1}_text': movie_df.iloc[movie_idx]['chunk_text'],
                f'match_{k+1}_similarity': similarity
            }
            generated_info.update(match_info)
        
        results.append(generated_info)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('chunk_similarities.csv', index=False)
    print("\nSimilarity results saved to chunk_similarities.csv")
    
    # Create visualization
    print("\nGenerating PCA visualization...")
    visualize_embeddings(generated_embeddings, movie_embeddings, generated_df, movie_df)
    print("Visualization saved as embedding_visualization.png")
    
    # Print some statistics
    print("\nSummary Statistics:")
    print(f"Number of generated chunks analyzed: {len(generated_df)}")
    print(f"Number of original movie chunks compared: {len(movie_df)}")
    print(f"Average similarity to top match: {top_k_similarities[:, 0].mean():.3f}")
    
    # Print example matches
    print("\nExample Matches (first chunk):")
    print(f"\nGenerated Text (Genre: {results_df.iloc[0]['genre']}):")
    print(results_df.iloc[0]['generated_text'])
    print("\nTop Match (Similarity: {:.3f}):".format(results_df.iloc[0]['match_1_similarity']))
    print(results_df.iloc[0]['match_1_text'])

if __name__ == "__main__":
    main() 