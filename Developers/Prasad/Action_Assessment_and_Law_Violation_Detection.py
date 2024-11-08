import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv("/content/Constitution Of India.csv")
df.head()

nltk.download('punkt')

texts = df['Articles'].tolist()
texts

tokenized_texts = [word_tokenize(text) for text in texts]
tokenized_texts

model_w2v = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

def get_avg_vector(text, model):
    words = word_tokenize(text)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

input_sentence = "he stayed in india without having citizenship of inda"
input_vector = get_avg_vector(input_sentence, model_w2v)

similarities = []
for text in texts:
    vector = get_avg_vector(text, model_w2v)
    similarity = cosine_similarity([input_vector], [vector])
    similarities.append(similarity[0][0])

most_similar_index = np.argmax(similarities)
most_similar_text = texts[most_similar_index]

print(f'Most similar text: {most_similar_text}')

from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

input_ids = tokenizer(input_sentence, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=32)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text

prompt_tuning = "Your role is to take the retrived data on the input_sentence and convert that retrived data as input_sentence suggest and answer which law has been violated in detail"

input_text = prompt_tuning + input_sentence + most_similar_text
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text = generated_text[len(input_text):].strip()
generated_text

def RAG_feature_3(input_text):
    input_vector = get_avg_vector(input_text, model_w2v)
    for text in texts:
        vector = get_avg_vector(text, model_w2v)
        similarity = cosine_similarity([input_vector], [vector])
        similarities.append(similarity[0][0])

    most_similar_index = np.argmax(similarities)
    most_similar_text = texts[most_similar_index]
    input_text = prompt_tuning + input_sentence + most_similar_text
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=200)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(input_text):].strip()
    return generated_text