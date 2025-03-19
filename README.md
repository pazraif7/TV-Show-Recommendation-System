This project is an AI-powered TV show recommendation system that helps users find new shows based on their favorites.
By entering a list of TV shows they enjoyed, users receive a list of similar shows along with two custom-generated show ideas, complete with AI-generated descriptions and cover images.
The system works by matching user input with a dataset of TV shows using fuzzy matching. 
It then generates text embeddings with OpenAI's API and finds the closest matches using vector similarity search. 
Additionally, it creates unique show ideas using AI and generates their cover images through the LightX API. 
The project is built with Python and uses Pandas, NumPy, TheFuzz (FuzzyWuzzy), OpenAI, Usearch, and Requests.
