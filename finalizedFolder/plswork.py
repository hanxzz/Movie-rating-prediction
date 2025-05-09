import re
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

API_KEY = "edc0eb4"
HF_API_KEY = "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"


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
X = df['title']  # Just using titles as dummy features for prediction
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

# Hugging Face inference section
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

script_text = """test"""  # Replace with actual script text

rating_prompt = (
    f"Read the script and give an honest rating BETWEEN 1.0 AND 10.0, return only the number. "
    f"You can also use decimal values for example: 1.3, 3.8, etc.\n\nScript: {script_text}\n\nRating:"
)

improvement_prompt = (
    f"Give three concise suggestions to improve this movie script in terms of dialogue, pacing, and narrative. "
    f"Answer in bullet points.\n\nScript: {script_text}\n\nSuggestions:"
)

# Get Rating from Hugging Face
rating = "N/A"
if script_text.strip():
    rating_response = requests.post(API_URL, headers=HEADERS, json={
        "inputs": rating_prompt,
        "parameters": {"max_new_tokens": 5}
    })

    if rating_response.status_code == 200 and rating_response.text.strip():
        try:
            rating_json = rating_response.json()
            rating_text = rating_json[0]["generated_text"]
            matches = re.findall(r'\d+\.\d+|\d+', rating_text)
            rating = matches[-1] if matches else "N/A"
        except Exception as e:
            print("Error parsing rating response:", e)
            rating = "N/A"
    else:
        print("Failed to get rating from Hugging Face:", rating_response.status_code)
        print(rating_response.text)

# Get Improvements from Hugging Face
improvements = "N/A"
if script_text.strip():
    improvement_response = requests.post(API_URL, headers=HEADERS, json={
        "inputs": improvement_prompt,
        "parameters": {"max_new_tokens": 150}
    })

    if improvement_response.status_code == 200 and improvement_response.text.strip():
        try:
            improvement_json = improvement_response.json()
            improvements = improvement_json[0]["generated_text"].replace(improvement_prompt, "").strip()
        except Exception as e:
            print("Error parsing improvement response:", e)
            improvements = "N/A"
    else:
        print("Failed to get improvements from Hugging Face:", improvement_response.status_code)
        print(improvement_response.text)

# Display output
print("\n------------- SCRIPT ---------------")
print(script_text)
print("------------------------------------")
print("\n---------- MOVIE DATA --------------")
print(df)
print("------------------------------------")
print(f"Train R-squared: {train_score:.4f}")
print(f"Test R-squared: {test_score:.4f}")
print("------------------------------------")
print("Predicted Rating from Script:", rating)
print("------------------------------------")
print("Improvement Suggestions:")
print(improvements)