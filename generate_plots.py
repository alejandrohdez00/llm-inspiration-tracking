import pandas as pd
from openai import OpenAI
import backoff
from openai import RateLimitError
from tqdm import tqdm
import random
import argparse

@backoff.on_exception(backoff.expo, RateLimitError)
def chat_completion_with_backoff(client, messages, **kwargs):
    """
    Wrapper function for chat completions with exponential backoff retry logic
    """
    return client.chat.completions.create(
        messages=messages,
        **kwargs
    )

def generate_plot(client: OpenAI, genre: str = None) -> str:
    """
    Generate a new movie plot using GPT-4
    """
    system_prompt = """You are a creative screenwriter who specializes in creating unique and engaging movie plots. 
Your plots should be original, compelling, and well-structured."""
    
    user_prompt = """Generate a unique and creative movie plot summary that is 3-4 paragraphs long. 
The plot should be coherent, engaging, and have a clear narrative arc with interesting character development."""
    
    if genre:
        user_prompt += f"\nThe movie should be in the {genre} genre."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = chat_completion_with_backoff(
        client,
        messages=messages,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content.strip()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate movie plots using GPT-4')
    parser.add_argument('--num_plots', type=int, default=10, 
                      help='Number of plots to generate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Optional: List of genres to use
    genres = [
        "Science Fiction", "Drama", "Thriller", "Comedy", "Fantasy",
        "Horror", "Romance", "Action", "Mystery", "Adventure"
    ]
    
    generated_plots = []
    
    print(f"Generating {args.num_plots} plots...")
    for i in tqdm(range(args.num_plots)):
        # Randomly select a genre
        genre = random.choice(genres)
        
        try:
            new_plot = generate_plot(client, genre)
            generated_plots.append({
                'plot_id': i,
                'genre': genre,
                'plot': new_plot
            })
        except Exception as e:
            print(f"Error generating plot {i}: {e}")
    
    # Save results
    output_df = pd.DataFrame(generated_plots)
    output_file = f"generated_plots_{args.num_plots}.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated {len(generated_plots)} plots. Results saved to {output_file}")
    
    # Print a sample plot
    if generated_plots:
        print("\nSample generated plot:")
        print(f"Genre: {generated_plots[0]['genre']}")
        print(f"Plot: {generated_plots[0]['plot']}")

if __name__ == "__main__":
    main() 