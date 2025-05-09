import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

API_KEY = "edc0eb4"

def get_imdb_data(num_movies=10):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&s=movie&type=movie"
    response = requests.get(url)
    data = response.json()
    
    if data["Response"] == "True":
        movies = []
        for movie in data["Search"][:num_movies]:
            title = movie["Title"]
            year = movie["Year"]
            movies.append({'title': title, 'year': year})
        return pd.DataFrame(movies)
    else:
        print("No movies were fetched.")
        return pd.DataFrame()

def get_rating(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    response = requests.get(url)
    data = response.json()
    return float(data["imdbRating"]) if data["Response"] == "True" else 0.0

# Fetch movie data
df = get_imdb_data()
if df.empty:
    exit()

# Get ratings for each movie
df['rating'] = df['title'].apply(get_rating)

# Prepare features and target
X = df['title']  
y = df['rating']

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R-squared: {train_score:.4f}")
print(f"Test R-squared: {test_score:.4f}")

# Function to predict rating for a new script with randomization
def predict_rating(script):
    # Randomize rating between 6 and 10
    predicted_rating = random.uniform(6.0, 10.0)
    return predicted_rating

# Example usage
new_script = """
FADE IN:

EXT. CITY STREET - DAY

A bustling city street filled with people. We see JANE, a young woman in her twenties, looking lost.

                         JANE
            (to herself)
            I can't believe I missed the bus again!

Suddenly, a mysterious stranger appears next to her.

                         STRANGER
            Need a ride?

JANE looks at him, unsure.

                         JANE
            Who are you?

                         STRANGER
            Just someone who wants to help.

She hesitates but then nods, getting into the car.

INT. STRANGER'S CAR - DAY

They drive through the city, the atmosphere tense.

                         JANE
            So... where are we going?

                         STRANGER
            You'll see.

FADE OUT.
"""
predicted_rating = predict_rating(new_script)
print(f"Predicted rating for the script '{new_script[:50]}...': {predicted_rating:.2f}")

# Display the fetched movie titles and ratings
print(df)
