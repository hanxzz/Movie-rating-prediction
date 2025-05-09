import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re

# Ensure NLTK data is downloaded
nltk.download('punkt')

# List of movie titles
movies = ["The Godfather", "Pulp Fiction", "Inception", "The Dark Knight", "Forrest Gump"]

# Updated script URLs
url_template = "https://imsdb.com/scripts/{}.html"

def fetch_script(title):
    url = url_template.format(title.replace(' ', '%20'))
    print(f"Fetching URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        script_div = soup.find('div', {'class': 'script'})
        if script_div:
            return script_div.get_text()
        else:
            print(f"Script for {title} not found.")
            return None
    else:
        print(f"Failed to fetch {title}. Status code: {response.status_code}")
        return None

def preprocess_script(script):
    # Basic preprocessing
    script = re.sub(r'\s+', ' ', script)  # Replace multiple whitespace with a single space
    return ' '.join(nltk.word_tokenize(script.lower()))

def main():
    scripts = []
    for movie in movies:
        print(f"Processing: {movie}")
        script = fetch_script(movie)
        if script:
            scripts.append(preprocess_script(script))

    if scripts:
        # Create and train the model
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(scripts, [1] * len(scripts))  # Dummy labels for training
        print("Model trained successfully.")
    else:
        print("No data to train on.")

if __name__ == "__main__":
    main()
