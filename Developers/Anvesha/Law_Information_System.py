#Law Information System CSI MAJOR PROJECT 1
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from IPython.display import display, HTML


import kagglehub

# Download latest version
path = kagglehub.dataset_download("arshid/iris-flower-dataset")

print("Path to dataset files:", path)

# Sample dataset (you can replace this with your own legal dataset)
import pandas as pd

# Replace 'data.csv' with the correct path to the file
data = pd.read_csv('Constitution Of India.csv')

# Convert to a DataFrame
df = pd.DataFrame(data)

# Display the first few entries of the dataset
df.head()


# Function to clean and preprocess the text
def preprocess_text(text):
    # Check if text is a string
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'\W', ' ', text)
        # Remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Convert to lowercase
        text = text.lower()
    else:
        # Handle non-string values (e.g., convert to empty string)
        text = ''
    return text

# Apply preprocessing to the 'case_text' column
df['Processed_Text'] = df['Articles'].apply(preprocess_text)

# Display the processed data
df.head()

# Initialize the TF-IDF vectorizer
#convert text into vectors
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the processed legal text
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Text'])


# Function to get similar documents based on user query
def search_laws(query):
    # Preprocess the query
    query_processed = preprocess_text(query)
    # Convert query into a vector
    query_vector = tfidf_vectorizer.transform([query_processed])
    # Calculate cosine similarity between the query and the documents
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Get top 3 most similar documents
    top_indices = np.argsort(similarities)[-3:][::-1]

    # Display the results
    print("\nTop Matching Laws:")
    for idx in top_indices:
        if similarities[idx] > 0:  # Display only if similarity score is positive
            print(f"\nText: {df.iloc[idx]['Articles']}\nSimilarity Score: {similarities[idx]:.2f}\n")

# Test the search function
search_laws("murder laws")

import ipywidgets as widgets
from IPython.display import display

# Create a text input field
text_input = widgets.Text(
    value='',
    placeholder='Enter your query here',
    description='Search:',
    disabled=False
)




# Function to handle input and display results
def handle_input(query):
    if query:
        search_laws(query)
    else:
        print("Please enter a search term.")

# Button to trigger search
button = widgets.Button(description="Search Laws")
button.on_click(lambda b: handle_input(text_input.value))

# Display the search box and button
display(text_input, button)

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from IPython.display import display, HTML
from google.colab import files
from transformers import pipeline
import ipywidgets as widgets

df['Processed_Text'] = df['Articles'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Text'])

generator = pipeline('text-generation', model='gpt2')

def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = text.lower()
    else:
        text = ''
    return text

def search_laws_with_generation(query):
    query_processed = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-3:][::-1]

    print("\nTop Matching Laws:")
    for idx in top_indices:
        if similarities[idx] > 0:
            print(f"\nText: {df.iloc[idx]['Articles']}\nSimilarity Score: {similarities[idx]:.2f}\n")

    # Generate a response based on the query
    generated_text = generator(query, max_length=100, num_return_sequences=1)
    print("\nGenerated Response based on your query:")
    print(generated_text[0]['generated_text'])


# Set up the search input widget
text_input = widgets.Text(
    value='',
    placeholder='Enter your query here',
    description='Search:',
    disabled=False
)


# Button to trigger search
button = widgets.Button(description="Search Laws")
button.on_click(lambda b: search_laws_with_generation(text_input.value))

# Display the search box and button
display(text_input, button)

