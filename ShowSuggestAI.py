from thefuzz import process
import pandas as pd
import os
import openai
import pickle
import numpy as np
import requests
import time
from PIL import Image
from usearch.index import Index

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
    titles = list(embeddings.keys())
    vectors = np.array(list(embeddings.values())).astype('float32')
    dim = vectors.shape[1]
    index = Index(ndim=dim)
    for idx, vector in enumerate(vectors):
        index.add(idx, vector)
    matches = index.search(average_vector, top_n)
    results_with_percentages = []
    for match in matches:
        title = titles[match.key]
        similarity = 1 - match.distance  
        percentage = min(100, round(similarity * 100))
        results_with_percentages.append((title, similarity, percentage))
    return results_with_percentages

def generate_show_name(base_shows, is_input_based):
    word_bank = ["Chronicles", "Tales", "Legacy", "Secrets", "Mystery", "Adventures"]
    base = base_shows[0].split()[0] if base_shows else "Mystery"
    suffix = word_bank[hash(str(base_shows)) % len(word_bank)]
    return f"The {base} {suffix}"

def generate_show_description(base_shows, is_input_based):
    if is_input_based:
        return f"A thrilling series that combines elements of {' and '.join(base_shows[:2])} into an original storyline"
    return f"A unique show that builds upon the themes of {base_shows[0]}, taking viewers on an unexpected journey"


def generate_lightx_image(show_name, show_description, output_file):
    try:
        api_key = os.getenv("LIGHTX_API_KEY")
        if not api_key:
            raise EnvironmentError("LIGHTX_API_KEY not found. Set it as an environment variable.")
        url = 'https://api.lightxeditor.com/external/api/v1/text2image'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        data = {
            "textPrompt": f'{show_description}',
            'n': 1,
            'size': '1024x1024'
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        data = response.json()

        if data["statusCode"] == 2000 and data["body"]["status"] == "init":
            order_id = data["body"]["orderId"]
            print(f"Request initiated successfully. Order ID: {order_id}")
            payload = {
                "orderId": f'{order_id}'
            }
            url2 = 'https://api.lightxeditor.com/external/api/v1/order-status'
            max_retries = data["body"].get("maxRetriesAllowed", 5)
            avg_response_time = data["body"].get("avgResponseTimeInSec", 2)
            for _ in range(max_retries):
                time.sleep(avg_response_time)  # Wait between retries
                status_response = requests.post(
                    url2,
                    headers=headers,
                    json=payload
                )
                status_response.raise_for_status()
                data_response = status_response.json()
                status = data_response["body"]["status"]
                if status == "active":
                    print("Image generation completed.")
                    image_url = data_response["body"]["output"]
                    img_data = requests.get(image_url).content
                    with open(output_file, 'wb') as f:
                        f.write(img_data)
                    return
                elif status == "failed":
                    raise Exception("Image generation failed.")
            raise TimeoutError("Image generation timed out.")
        else:
            raise Exception(f"Unexpected response: {data}")

    except requests.exceptions.RequestException as e:
        print(f"Error generating image: {e}")
        return None
    
def create_custom_shows(matched_shows, filtered_recommendations):
    show1name = generate_show_name(matched_shows, is_input_based=True)
    show1description = generate_show_description(matched_shows, is_input_based=True)
    recommended_titles = [rec[0] for rec in filtered_recommendations]
    show2name = generate_show_name(recommended_titles, is_input_based=False)
    show2description = generate_show_description(recommended_titles, is_input_based=False)
    generate_lightx_image(show1name, show1description, "show1.png")
    generate_lightx_image(show2name, show2description, "show2.png")
    print("I have also created just for you two shows which I think you would love.")
    print(f"Show #1 is based on the fact that you loved the input shows that you gave me. Its name is {show1name} and it is about {show1description}.")
    print(f"Show #2 is based on the shows that I recommended for you. Its name is {show2name} and it is about {show2description}.")
    print("Here are also the 2 tv show ads. Hope you like them!")
    Image.open("show1.png").show()
    Image.open("show2.png").show()

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

        create_custom_shows(matched_shows, filtered_recommendations)
    
        break
       

if __name__ == "__main__":
    main()





