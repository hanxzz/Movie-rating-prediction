import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# OMDB API Key
API_KEY = "edc0eb4"

# Function to fetch detailed movie data
def get_movie_details(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    response = requests.get(url)
    return response.json()

# Function to fetch a list of movies
def get_imdb_data(num_movies=10):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&s=movie&type=movie"
    response = requests.get(url)
    data = response.json()
    
    if data["Response"] == "True":
        movies = []
        for movie in data["Search"][:num_movies]:
            title = movie["Title"]
            year = movie["Year"]
            movie_data = get_movie_details(title)
            if movie_data["Response"] == "True" and "imdbRating" in movie_data:
                movies.append({
                    'title': title,
                    'year': year,
                    'genre': movie_data.get('Genre', ''),
                    'plot': movie_data.get('Plot', ''),
                    'rating': float(movie_data.get('imdbRating', 0.0))
                })
        return pd.DataFrame(movies)
    else:
        print("No movies were fetched.")
        return pd.DataFrame()

# Fetch movie data
df = get_imdb_data(num_movies=10)
if df.empty:
    print("Failed to fetch movie data. Exiting...")
    exit()

# Combine relevant features for TF-IDF
df['combined_features'] = df['title'] + " " + df['genre'] + " " + df['plot']

# Prepare features and target
X = df['combined_features']
y = df['rating']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R-squared: {train_score:.4f}")
print(f"Test R-squared: {test_score:.4f}")

# Function to predict rating for a new script
def predict_rating(script):
    script_vector = vectorizer.transform([script])
    predicted_rating = model.predict(script_vector)[0]
    return predicted_rating

# Example usage
new_script_1 = """
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

new_script_2 = """
FADE IN:

INT. SPACESHIP - DEEP SPACE

Captain Reynolds sits in his command chair, staring out at the stars.

                         REYNOLDS
            We've come so far... but at what cost?

The crew works silently, each person lost in their thoughts. Suddenly, alarms blare.

                         CREWMATE
            Captain! Enemy ships detected!

Reynolds stands, determination in his eyes.

                         REYNOLDS
            Battle stations, everyone. We fight to survive.

FADE OUT.
"""

# Predict ratings for scripts
predicted_rating_1 = predict_rating(new_script_1)
predicted_rating_2 = predict_rating(new_script_2)

print(f"Predicted rating for script 1: {predicted_rating_1:.2f}")
print(f"Predicted rating for script 2: {predicted_rating_2:.2f}")

# Display the movie data
print(df[['title', 'genre', 'rating']])
