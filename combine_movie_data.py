import csv
import json
import pandas as pd

def load_metadata(filename):
    movies = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                fields = line.strip().split('\t')
                movie_id = fields[0]
                
                # Parse the genres dictionary string into a list of genres
                genres_dict = eval(fields[8])  # Using eval to parse the dictionary string
                genres = list(genres_dict.values())
                
                # Parse the language dictionary
                lang_dict = eval(fields[6])
                languages = list(lang_dict.values())
                
                # Parse the country dictionary
                country_dict = eval(fields[7])
                countries = list(country_dict.values())
                
                movies[movie_id] = {
                    'movie_id': movie_id,
                    'freebase_id': fields[1],
                    'title': fields[2],
                    'release_date': fields[3],
                    'revenue': fields[4],
                    'runtime': fields[5],
                    'languages': ','.join(languages),
                    'countries': ','.join(countries),
                    'genres': ','.join(genres)
                }
            except Exception as e:
                print(f"Error processing movie {movie_id}: {e}")
                continue
    return movies

def load_plots(filename):
    plots = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                movie_id, plot = line.strip().split('\t')
                # Clean the plot text of any problematic characters
                plot = plot.encode('utf-8', errors='ignore').decode('utf-8')
                plots[movie_id] = plot
            except Exception as e:
                print(f"Error processing plot for movie {movie_id}: {e}")
                continue
    return plots

def create_combined_csv(metadata_file, plots_file, output_file='movie_data_combined.csv'):
    # Load both data sources
    print("Loading metadata...")
    movies = load_metadata(metadata_file)
    print("Loading plots...")
    plots = load_plots(plots_file)
    
    # Prepare the CSV file
    headers = ['movie_id', 'freebase_id', 'title', 'release_date', 'revenue', 
              'runtime', 'languages', 'countries', 'genres', 'plot']
    
    print(f"Writing combined data to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        # Combine and write the data
        for movie_id in movies:
            try:
                row = movies[movie_id].copy()
                plot = plots.get(movie_id, '')
                # Clean the plot text again before writing
                row['plot'] = plot.encode('utf-8', errors='ignore').decode('utf-8')
                writer.writerow(row)
            except Exception as e:
                print(f"Error writing movie {movie_id}: {e}")
                continue
    
    print("Data combination complete!")
    
    # Show a sample of the combined data
    print("\nFirst few rows of the combined data:")
    return pd.read_csv(output_file, nrows=5)

if __name__ == "__main__":
    metadata_file = 'data-summaries/MovieSummaries/movie.metadata.tsv'
    plots_file = 'data-summaries/MovieSummaries/plot_summaries.txt'
    
    try:
        sample_data = create_combined_csv(metadata_file, plots_file)
        print(sample_data)
    except Exception as e:
        print(f"An error occurred: {e}")
    