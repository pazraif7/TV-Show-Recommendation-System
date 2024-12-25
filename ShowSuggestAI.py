from thefuzz import process
import pandas as pd
import os
import openai
import pickle
import numpy as np
from scipy.spatial.distance import cosine

def match_shows(user_input, available_shows):
    data = pd.read_csv("imdb_tvshows - imdb_tvshows.csv")
    available_shows = data['Title'].tolist()
    matched = []
    for show in user_input:
        match = process.extractOne(show, available_shows)  
        if match and match[1] >= 80:  
            matched.append(match[0])
    return matched


def generate_embeddings(tvshows_file="imdb_tvshows - imdb_tvshows.csv", output_folder="embeddings_pickle"):
    pickle_file = os.path.join(output_folder, "embeddings.pkl")
    if os.path.exists(pickle_file):
        print(f"Embeddings file found at {pickle_file}. Loading embeddings...")
        with open(pickle_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Total TV shows in embeddings: {len(embeddings)}")
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


def average_vector(matched_shows, embeddings):
    vectors = []
    for show in matched_shows:
        if show in embeddings:
            vectors.append(np.array(embeddings[show]))
        else:
            print(f"Warning: Embedding not found for '{show}'")
    if not vectors:
        print("No embeddings found for matched shows.")
        return None
    return np.mean(vectors, axis=0)


def find_closest_shows(average_vector, embeddings, top_n=5):
    similarities = []
    for title, vector in embeddings.items():
        similarity = 1 - cosine(average_vector, vector)  
        similarities.append((title, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    max_similarity = similarities[0][1] if similarities else 1
    results_with_percentages = [
        (title, similarity, (similarity / max_similarity) * 100)
        for title, similarity in similarities
    ]
    return results_with_percentages[:top_n]

def main():
    print("Welcome to the TV Show Recommendation System!")
    embeddings = generate_embeddings()
    while True:
        user_input = input(
            "Which TV shows did you really like watching? Separate them by a '/'.\n"
            "Make sure to enter more than 1 show: "
        ).strip()
        
        if not user_input:
            print("Please enter at least two TV shows.")
            continue

        user_shows = [show.strip() for show in user_input.split("/") if show.strip()]
        if len(user_shows) < 2:
            print("Please enter more than 1 show.")
            continue

        matched_shows = match_shows(user_shows, available_shows=None)
        if not matched_shows:
            print("Sorry, no matches found for the TV shows you entered. Try again.")
            continue

        matched_shows_str = " / ".join(matched_shows)
        confirmation = input(
            f"Making sure, do you mean {matched_shows_str}? (y/n): "
        ).strip().lower()

        if confirmation != "y":
            print("Sorry about that. Let's try again. Please make sure to write the names of the TV shows correctly.")
            continue

        print("Great! Generating recommendations now…")

        avg_vector = average_vector(matched_shows, embeddings)
        if avg_vector is None:
            print("Sorry, we couldn't find embeddings for the shows you entered. Try again.")
            continue

        recommendations = find_closest_shows(avg_vector, embeddings, top_n=10)
        filtered_recommendations = [
            (title, similarity, percentage)
            for title, similarity, percentage in recommendations if title not in matched_shows
        ][:5]

        print("Here are the TV shows that I think you would love:")
        for title, _, percentage in filtered_recommendations:
            print(f"- {title}: {percentage:.2f}% match")
        break

if __name__ == "__main__":
    main()





