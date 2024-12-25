from thefuzz import process
import pandas as pd

def match_shows(user_input, available_shows):
    data = pd.read_csv("imdb_tvshows - imdb_tvshows.csv")
    available_shows = data['Title'].tolist()
    matched = []
    for show in user_input:
        match = process.extractOne(show, available_shows)
        print(f"Input: {show}, Match: {match}")  # Debugging output
        if match and match[1] >= 80:  # Threshold of 80
            matched.append(match[0])
    return matched
