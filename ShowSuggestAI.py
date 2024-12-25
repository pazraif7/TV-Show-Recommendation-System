from thefuzz import process
import pandas as pd
import os
import openai
import pickle

def match_shows(user_input, available_shows):
    data = pd.read_csv("imdb_tvshows - imdb_tvshows.csv")
    available_shows = data['Title'].tolist()
    matched = []
    for show in user_input:
        match = process.extractOne(show, available_shows)
        print(f"Input: {show}, Match: {match}")  
        if match and match[1] >= 80:  
            matched.append(match[0])
    return matched


def generate_embeddings(tvshows_file="imdb_tvshows - imdb_tvshows.csv", output_folder="embeddings_pickle"):
    pickle_file = os.path.join(output_folder, "embeddings.pkl")
    if os.path.exists(pickle_file):
        print(f"Embeddings file found at {pickle_file}. Loading embeddings...")
        with open(pickle_file, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Set it as an environment variable.")

    data = pd.read_csv(tvshows_file)
    embeddings = {}
    print("Starting to generate embeddings...")
    for idx, row in data.iterrows():
        title = row['Title']
        description = row['Description']
        print(f"Generating embedding for: {title}")
        
        try:
            response = openai.Embedding.create(
                input=description,
                model="text-embedding-ada-002"
            )
            embeddings[title] = response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding for {title}: {e}")

    with open(pickle_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings successfully saved to {pickle_file}")
    return embeddings

