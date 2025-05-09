import requests
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": "Bearer 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"}


# Function to get movie rating
def get_script_rating(script):
    if not script.strip():
        return "N/A"

    prompt = (f"Read the script and give an honest rating BETWEEN 1.0 AND 10.0, "
              f"return only the number without any extra text.\n\nScript: {script}\n\nRating:")

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt, "parameters": {"max_new_tokens": 5}})
    rating_text = response.json()[0]["generated_text"]

    matches = re.findall(r'\d+\.\d+|\d+', rating_text)
    return matches[-1] if matches else "N/A"


# Function to get improvement suggestions
def get_script_suggestions(script):
    if not script.strip():
        return "No script provided."

    prompt = (
        f"Give three concise suggestions to improve this movie script in terms of dialogue, pacing, and narrative. "
        f"Answer in bullet points.\n\nScript: {script}\n\nSuggestions:")

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt, "parameters": {"max_new_tokens": 150}})
    return response.json()[0]["generated_text"].replace(prompt, "").strip()


# Function to predict rating
def predict_rating():
    script = text_area.get("1.0", tk.END).strip()
    if not script:
        messagebox.showwarning("Input Error", "Please enter a script.")
        return

    rating = get_script_rating(script)
    result_label.config(text=f"Predicted Rating: {rating}")


# Function to show suggestions
def show_suggestions():
    script = text_area.get("1.0", tk.END).strip()
    if not script:
        messagebox.showwarning("Input Error", "Please enter a script.")
        return

    suggestions = get_script_suggestions(script)

    suggestions_window = tk.Toplevel(root)
    suggestions_window.title("Improvement Suggestions")

    text_widget = scrolledtext.ScrolledText(suggestions_window, wrap=tk.WORD, width=60, height=10)
    text_widget.pack(padx=10, pady=10)
    text_widget.insert(tk.END, suggestions)
    text_widget.config(state="disabled")


# GUI Setup
root = tk.Tk()
root.title("Movie Rating Predictor")

label = tk.Label(root, text="Enter your movie script below:")
label.pack()

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
text_area.pack(padx=10, pady=10)

predict_button = tk.Button(root, text="Predict Rating", command=predict_rating)
predict_button.pack(pady=5)

suggestions_button = tk.Button(root, text="Show Suggestions", command=show_suggestions)
suggestions_button.pack(pady=5)

result_label = tk.Label(root, text="Predicted Rating: N/A", font=("Helvetica", 14))
result_label.pack(pady=5)

root.mainloop()
