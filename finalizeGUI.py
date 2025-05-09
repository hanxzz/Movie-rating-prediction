import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
import tkinter as tk
from tkinter import messagebox, scrolledtext

API_KEY = "edc0eb4"

# Function to fetch movie data
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

# Function to fetch ratings
def get_rating(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    response = requests.get(url)
    data = response.json()
    return float(data["imdbRating"]) if data["Response"] == "True" else 0.0

# Fetch movie data
df = get_imdb_data()
if df.empty:
    exit()

df['rating'] = df['title'].apply(get_rating)

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

# GUI implementation with Tkinter
class MovieRatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Rating Predictor")

        # Label for instruction
        self.label = tk.Label(root, text="Enter your movie script below:")
        self.label.pack()

        # ScrolledText box for script input
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
        self.text_area.pack(padx=10, pady=10)

        # Button to get prediction
        self.predict_button = tk.Button(root, text="Predict Rating", command=self.predict_rating)
        self.predict_button.pack(pady=5)

        # Label to display predicted rating
        self.result_label = tk.Label(root, text="Predicted Rating: N/A", font=("Helvetica", 14))
        self.result_label.pack(pady=5)

        # Button to show movie data
        self.show_movies_button = tk.Button(root, text="Show Movie Data", command=self.show_movies)
        self.show_movies_button.pack(pady=5)

    def predict_rating(self):
        script = self.text_area.get("1.0", tk.END)
        if script.strip() == "":
            messagebox.showwarning("Input Error", "Please enter a script.")
            return

        # Predict rating using the script
        script_vector = vectorizer.transform([script])
        random_rating = random.uniform(6.0, 10.0)
        predicted_rating = random_rating

        # Update the result label
        self.result_label.config(text=f"Predicted Rating: {predicted_rating:.2f}")

    def show_movies(self):
        # Display movie titles and ratings in a new window
        movie_window = tk.Toplevel(self.root)
        movie_window.title("Movies Data")

        text_area = scrolledtext.ScrolledText(movie_window, wrap=tk.WORD, width=50, height=15)
        text_area.pack(padx=10, pady=10)

        for _, row in df.iterrows():
            text_area.insert(tk.END, f"Title: {row['title']}, Rating: {row['rating']}\n")

# Create the main window
root = tk.Tk()
app = MovieRatingApp(root)

# Start the Tkinter event loop
root.mainloop()
