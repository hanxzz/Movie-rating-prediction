import requests
import asyncio
import aiohttp  # For asynchronous HTTP requests
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')

API_KEY = "edc0eb4"  # OMDb API key

# List of movie titles
movies = ["The Godfather", "Pulp Fiction", "Inception", "The Dark Knight", "Forrest Gump"]

async def fetch_omdb_data(session, title):
    """Fetch movie information (including the plot) from OMDb API asynchronously."""
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            if data['Response'] == 'True':
                return data
            else:
                print(f"OMDb Error: {data['Error']}")
                return None
        else:
            print(f"Failed to fetch {title} from OMDb. Status code: {response.status}")
            return None

async def fetch_movie_data():
    """Fetch movie data concurrently using aiohttp."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_omdb_data(session, movie) for movie in movies]
        return await asyncio.gather(*tasks)

def preprocess_script(script):
    # Basic preprocessing
    script = re.sub(r'\s+', ' ', script)  # Replace multiple whitespace with a single space
    try:
        return ' '.join(nltk.word_tokenize(script.lower()))
    except LookupError as e:
        print(f"Error in tokenization: {e}")
        return ""

async def main():
    scripts = []
    ratings = []

    movie_data_list = await fetch_movie_data()

    for movie_data in movie_data_list:
        if movie_data:
            title = movie_data['Title']
            plot = movie_data.get('Plot', '')  # Use the movie plot as the "script"
            rating = float(movie_data.get('imdbRating', 0))
            print(f"Movie: {title}, IMDB Rating: {rating}, Plot: {plot[:50]}...")

            if plot:
                processed_script = preprocess_script(plot)
                scripts.append(processed_script)
                ratings.append(rating)

    if scripts and ratings:
        # Create and train the model
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(scripts, ratings)  # Train with the actual IMDb ratings
        print("Model trained successfully.")
    else:
        print("No data to train on.")

if __name__ == "__main__":
    asyncio.run(main())
