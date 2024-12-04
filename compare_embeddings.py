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
import plotly.express as px
import plotly.graph_objects as go

def get_main_genre(genres_str: str, target_genres: List[str]) -> str:
    """
    Get the first genre that matches our target genres list.
    If no match is found, return 'Other'
    """
    if pd.isna(genres_str):
        return 'Other'
        
    genres = genres_str.split(',')
    for genre in genres:
        genre = genre.strip()
        if genre in target_genres:
            return genre
    return 'Other'

def load_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load the embeddings and metadata for both generated and original plots
    """
    # Define target genres (same as in generate_plots.py)
    target_genres = [
        "Science Fiction", "Drama", "Thriller", "Comedy", "Fantasy",
        "Horror", "Romance", "Action", "Mystery", "Adventure"
    ]
    
    # Load embeddings
    generated_embeddings = np.load("generated_plots_10_embeddings.npy")
    movie_embeddings = np.load("movie_chunks_embeddings.npy")
    
    # Load metadata - make sure to load the chunk-level data
    generated_df = pd.read_csv("generated_plots_10_embeddings.csv")
    movie_chunks_df = pd.read_csv("movie_chunks_embeddings.csv")
    
    # Load movie metadata and extract genres
    movie_metadata = pd.read_csv("movie_data_transformed_1500_plots.csv")
    
    # Create columns for main genre (first matching target genre) and all genres
    movie_metadata['main_genre'] = movie_metadata['genres'].apply(
        lambda x: get_main_genre(x, target_genres)
    )
    
    # Merge genres into chunks dataframe
    movie_chunks_df = movie_chunks_df.merge(
        movie_metadata[['movie_id', 'genres', 'main_genre']], 
        on='movie_id', 
        how='left'
    )
    
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
    
    Args:
        query_embeddings: embeddings of generated chunks
        reference_embeddings: embeddings of original movie chunks
        top_k: number of most similar chunks to return (default=5)
    
    Returns:
        top_k_similarities: array of shape (n_queries, top_k) with similarity scores
        top_k_indices: array of shape (n_queries, top_k) with indices of most similar chunks
    """
    # Convert to torch tensors for efficient computation
    query_tensor = torch.from_numpy(query_embeddings).float()
    ref_tensor = torch.from_numpy(reference_embeddings).float()
    
    # Normalize embeddings for cosine similarity
    query_normalized = F.normalize(query_tensor, p=2, dim=1)
    ref_normalized = F.normalize(ref_tensor, p=2, dim=1)
    
    # Compute similarities in batches to avoid memory issues
    batch_size = 1000
    all_similarities = []
    
    for i in tqdm(range(0, len(query_normalized), batch_size), desc="Computing similarities"):
        batch = query_normalized[i:i+batch_size]
        # Compute cosine similarity: dot product of normalized vectors
        sim = torch.mm(batch, ref_normalized.t())
        all_similarities.append(sim)
    
    # Combine all batches
    similarities = torch.cat(all_similarities, dim=0)
    
    # Get top-k similarities and indices
    top_k_similarities, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
    
    return top_k_similarities.numpy(), top_k_indices.numpy()

def visualize_embeddings(generated_embeddings: np.ndarray, 
                        movie_embeddings: np.ndarray,
                        generated_df: pd.DataFrame,
                        movie_df: pd.DataFrame):
    """
    Create both static (matplotlib) and interactive (plotly) visualizations of the embedding space
    """
    # Combine embeddings
    all_embeddings = np.vstack([generated_embeddings, movie_embeddings])
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(all_embeddings)
    
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)
    
    # Create DataFrame with all the data
    df_plot = pd.DataFrame()
    
    # Add generated chunks
    generated_plot_data = pd.DataFrame({
        'x': reduced_embeddings[:len(generated_embeddings), 0],
        'y': reduced_embeddings[:len(generated_embeddings), 1],
        'text': generated_df['chunk_text'],
        'type': 'Generated',
        'main_genre': generated_df['genre'],
        'all_genres': generated_df['genre']  # For generated, main and all genres are the same
    })
    
    # Add original movie chunks
    movie_plot_data = pd.DataFrame({
        'x': reduced_embeddings[len(generated_embeddings):, 0],
        'y': reduced_embeddings[len(generated_embeddings):, 1],
        'text': movie_df['chunk_text'],
        'type': 'Original',
        'main_genre': movie_df['main_genre'],
        'all_genres': movie_df['genres']
    })
    
    df_plot = pd.concat([generated_plot_data, movie_plot_data])
    
    # Create the interactive plot
    fig = px.scatter(
        df_plot, 
        x='x', 
        y='y',
        color='main_genre',  # Use main genre for coloring
        symbol='type',  # Use different symbols for Generated vs Original
        hover_data={
            'text': True, 
            'x': False, 
            'y': False, 
            'main_genre': False,  # Hide main_genre since it's shown in color
            'all_genres': True,
            'type': True
        },
        title='Interactive Document Embeddings Visualization',
        labels={'x': f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2%})',
                'y': f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2%})',
                'main_genre': 'Genre'}
    )

    # Update traces to add black border to generated chunks
    for trace in fig.data:
        # Get the type from the name of the trace (which includes the symbol info)
        is_generated = 'Generated' in trace.name
        trace.marker.line = dict(
            color='black' if is_generated else 'rgba(0,0,0,0)',
            width=2 if is_generated else 0
        )

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>Type:</b> %{customdata[2]}<br>" +
                     "<b>Genres:</b> %{customdata[1]}<br>" +
                     "<b>Text:</b> %{customdata[0]}<extra></extra>"
    )

    # Update layout
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        legend_title_text='Genre & Type'
    )

    # Save the interactive plot
    fig.write_html('embedding_visualization_interactive.html')

def save_similarity_report(results_df: pd.DataFrame, movie_df: pd.DataFrame, output_file: str = 'similarity_report.txt'):
    """
    Create a detailed report of similarities between generated and original chunks
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in results_df.iterrows():
            # Write generated chunk info
            f.write(f"\n{'='*100}\n")
            f.write(f"GENERATED CHUNK (Genre: {row['genre']})\n")
            f.write(f"{'-'*50}\n")
            f.write(f"{row['generated_text']}\n\n")
            
            # Write top 3 matches
            f.write("TOP 3 SIMILAR ORIGINAL CHUNKS:\n")
            for i in range(1, 4):  # top 3 matches
                movie_id = row[f'match_{i}_movie_id']
                movie_info = movie_df[movie_df['movie_id'] == movie_id].iloc[0]
                
                f.write(f"\n{i}. Similarity Score: {row[f'match_{i}_similarity']:.3f}\n")
                f.write(f"Movie ID: {movie_id}\n")
                f.write(f"Genres: {movie_info['genres']}\n")
                f.write(f"Chunk: {row[f'match_{i}_text']}\n")
                f.write(f"{'-'*50}\n")

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
    
    # Create detailed similarity report
    print("\nGenerating similarity report...")
    save_similarity_report(results_df, movie_df)
    print("Detailed similarity report saved as similarity_report.txt")
    
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