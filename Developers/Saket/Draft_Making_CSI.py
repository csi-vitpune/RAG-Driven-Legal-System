import numpy as np
import pandas as pd
import string
from textblob import TextBlob

#!/bin/bash
# !kaggle datasets download rushikeshdarge/constitution-of-india

# !unzip constitution-of-india.zip

df = pd.read_csv('/content/Constitution Of India.csv')

df.head()

df.isna()

df['Cleaned_articles'] = df['Articles'].str.lower()

df.head()

exclude = string.punctuation
print(exclude)

def remove_punc(text):
    return text.translate(str.maketrans(' ',' ',exclude))

df['Cleaned_articles'] = df['Cleaned_articles'].apply(remove_punc)

df['Cleaned_articles']

def spell_correct(text):
    return str(TextBlob(text).correct())

df['Cleaned_articles'] = df['Cleaned_articles'].apply(spell_correct)

df['Cleaned_articles']

'''!wget https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz
!tar -xvf en.tar.gz'''

#!pip install jamspell

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stopwords.words('english')

def remove_stopwords(text):
    stop_words= set(stopwords.words('english'))
    new_text = [word if word not in stop_words else '' for word in text.split()]
    return ' '.join(new_text).strip()

df['Cleaned_articles'] = df['Cleaned_articles'].apply(remove_stopwords)

df['Cleaned_articles']

import spacy
nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    # Check if the input is a list of characters and clean it up
    if isinstance(text, list):
        # Join the characters and remove excessive spaces
        text = ''.join(text).strip()  # Joins and strips leading/trailing spaces
    # Process the cleaned string with SpaCy for tokenization
    doc = nlp(text)
    # Extract tokens
    return [token.text for token in doc if token.text.strip()]

# Apply the tokenization function to the 'Articles' column
df['Cleaned_articles'] = df['Cleaned_articles'].apply(tokenize)


df['Cleaned_articles'][10]

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
l1=['running', 'ran', 'run']

load_model = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def lemmitize(text):
  text = ' '.join(text)
  doc = load_model(text)
  return  [token.lemma_ for token in doc]

df['Cleaned_articles'] = df['Cleaned_articles'].apply(lemmitize)

df['Cleaned_articles']

import gensim
from gensim.models import Word2Vec,KeyedVectors

# Assuming 'df' is your DataFrame and 'Articles' is the column containing the preprocessed text
sentence = df['Cleaned_articles'].tolist()  # Convert the 'Articles' column to a list of lists

model_wv = Word2Vec(sentence, vector_size=384, window=5, min_count=1, workers=4)

def word2vec(text):
  return model_wv.wv[text]



df['Cleaned_articles'] = df['Cleaned_articles'].apply(word2vec)

df['Cleaned_articles']

# !pip install chromadb

import chromadb

client = chromadb.Client()

collection = client.create_collection(name="articles")

for index, row in df.iterrows():
    article_vectors = row['Cleaned_articles']
    original_text=row['Articles']
    num_embeddings = len(article_vectors)


    ids = [f"article_{index}_embedding_{i}" for i in range(num_embeddings)]

    collection.add(
        embeddings=article_vectors,
        ids= ids,
        documents=[original_text]*num_embeddings,
        metadatas=[{"index": index, "embedding_id": i} for i in range(num_embeddings)]
    )

query_vector = df['Cleaned_articles'][0][0]
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
print(results)

# !pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

# !pip install accelerate

# !pip install sentence_transformers

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def word2vec_1(text):
    """
    Converts a list of tokens into a word embedding vector.
    Handles KeyError by ignoring words not in the vocabulary.
    """
    vectors = [model_wv.wv[token] for token in text if token in model_wv.wv]
    if vectors:  # Check if any valid vectors were found
        return np.mean(vectors, axis=0)  # Average the vectors
    else:
        return np.zeros(model_wv.vector_size)  # Return a zero vector if no words found



# Initialize tokenizer and models
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
model = GPT2LMHeadModel.from_pretrained("gpt2")
#model = model.to("cuda")

embedder = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=True)#.cuda()

prompt_tuning = """
Your task is to generate a detailed case draft. The draft should include the following sections:
1. Introduction
2. Statement of Facts
3. Jurisdiction
4. Claims
5. Prayer for Relief
"""

# Input case title and article
# print("Enter case title:")
# case_title = input()
# print("Enter facts:")
# facts = input()

query_in = f"Case Title: {case_title}, Facts Used: {facts}\n"
query_lower = query_in.lower()
query_remove_punc = remove_punc(query_lower)
query_tokenize = tokenize(query_remove_punc)
query_lemmitize = lemmitize(query_tokenize)
query_embedding = word2vec_1(query_lemmitize)

threshold = 0.5


matching = collection.query(query_embeddings=[query_embedding], include=['documents','distances'], n_results=10)

matching_docs = []
for i, distance in enumerate(matching['distances'][0]):
    if distance <= threshold:  # Only consider matches within the threshold
        matching_docs.append(matching['documents'][0][i])

# Check if we have any valid matching documents
if matching_docs:
    matching_text = matching_docs[0]  # Get the best match
else:
    matching_text = "No matching document found within the threshold."

# Debug: Print the matching text
print("Matching Text:", matching_text)

# Combine article and matching text
combined_text = facts + " " + matching_text  # Ensure there's a space for readability
max_combined_length = 768  # Leave enough space for prompt and query
truncated_combined_text = combined_text[:max_combined_length]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Create the input text
input_text = prompt_tuning + query_in + truncated_combined_text

# Tokenize input text
input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
input_ids = input_ids.input_ids#.to("cuda")
attention_mask = input_ids != tokenizer.pad_token_id
attention_mask = attention_mask.long()#.to("cuda")

# Generate output from the model
outputs = model.generate(
    input_ids,
    max_new_tokens=500,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0
)

# Decode the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)

def RAG_feature_4(input_text):
    query_in = f"Case Title: {case_title}, Facts Used: {facts}\n"
    query_lower = query_in.lower()
    query_remove_punc = remove_punc(query_lower)
    query_tokenize = tokenize(query_remove_punc)
    query_lemmitize = lemmitize(query_tokenize)
    query_embedding = word2vec_1(query_lemmitize)
    threshold = 0.5
    matching = collection.query(query_embeddings=[query_embedding], include=['documents','distances'], n_results=10)
    matching_docs = []
    for i, distance in enumerate(matching['distances'][0]):
        if distance <= threshold:  # Only consider matches within the threshold
            matching_docs.append(matching['documents'][0][i])
    if matching_docs:
        matching_text = matching_docs[0]  # Get the best match
    else:
        matching_text = "No matching document found within the threshold."
    combined_text = facts + " " + matching_text  # Ensure there's a space for readability
    max_combined_length = 768  # Leave enough space for prompt and query
    truncated_combined_text = combined_text[:max_combined_length]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Create the input text
    input_text = prompt_tuning + query_in + truncated_combined_text

    # Tokenize input text
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    input_ids = input_ids.input_ids#.to("cuda")
    attention_mask = input_ids != tokenizer.pad_token_id
    attention_mask = attention_mask.long()#.to("cuda")

    # Generate output from the model
    outputs = model.generate(
        input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )

    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text