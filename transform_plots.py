import pandas as pd
from openai import OpenAI
import time
from typing import Optional
import argparse
import os
from tqdm import tqdm
import backoff
from openai import RateLimitError

@backoff.on_exception(backoff.expo, RateLimitError)
def chat_completion_with_backoff(client, messages, **kwargs):
    """
    Wrapper function for chat completions with exponential backoff retry logic
    """
    return client.chat.completions.create(
        messages=messages,
        **kwargs
    )

def anonymize_and_chunk_plot(client, plot: Optional[str]) -> str:
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
        
        # Print usage information
        usage = response.usage
        print(f"\nTokens used in this request:")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")
        print(f"  Details: {usage.prompt_tokens_details}")
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error processing plot: {e}")
        return ""

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transform movie plots using GPT-4')
    parser.add_argument('--num_plots', type=int, default=None, 
                      help='Number of plots to transform. If not specified, transforms all plots.')
    args = parser.parse_args()

    client = OpenAI()

    # Read both the input CSV and previously transformed data
    df = pd.read_csv('movie_data_combined.csv')
    try:
        df_previous = pd.read_csv('movie_data_transformed_500_plots.csv')
        # Create a set of previously transformed movie_ids for faster lookup
        transformed_ids = set(df_previous['movie_id'].values)
        print(f"Found {len(transformed_ids)} previously transformed plots to skip")
    except FileNotFoundError:
        transformed_ids = set()
        print("No previously transformed plots found")
    
    # Create new column for transformed plots
    df['transformed_plot'] = ''
    
    # Process plots
    processed_count = 0
    idx = 0
    total_rows = len(df)
    
    # Initialize progress bar
    pbar = tqdm(total=args.num_plots if args.num_plots else total_rows, 
                desc="Transforming plots", 
                unit="plot")
    
    while (args.num_plots is None or processed_count < args.num_plots) and idx < total_rows:
        row = df.iloc[idx]
        
        # Skip if this movie was already transformed
        if row['movie_id'] in transformed_ids:
            print(f"\nSkipping movie {idx + 1} - already transformed")
            idx += 1
            continue
            
        if pd.isna(row['plot']) or not str(row['plot']).strip():
            print(f"\nSkipping movie {idx + 1} - empty plot")
            idx += 1
            continue
            
        print(f"\nProcessing movie {idx + 1} (Plot {processed_count + 1}" + 
              f"{f'/{args.num_plots}' if args.num_plots else ''})")
        
        transformed_plot = anonymize_and_chunk_plot(client, row['plot'])
        df.at[idx, 'transformed_plot'] = transformed_plot
        
        processed_count += 1
        idx += 1
        pbar.update(1)
    
    pbar.close()
    
    # Filter the DataFrame to keep only rows with non-empty transformed plots
    df_transformed = df[df['transformed_plot'].str.strip() != '']
    
    # Combine with previous transformations if they exist
    if len(transformed_ids) > 0:
        df_transformed = pd.concat([df_previous, df_transformed], ignore_index=True)
    
    # Save the filtered DataFrame
    output_filename = f'movie_data_transformed_{len(df_transformed)}_plots.csv'
    df_transformed.to_csv(output_filename, index=False)
    print(f"\nTransformation complete! Results saved to {output_filename}")
    print(f"Processed {processed_count} new plots")
    print(f"Total plots in output file: {len(df_transformed)}")

if __name__ == "__main__":
    main() 