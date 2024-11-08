import numpy as np
import pandas as pd

df = pd.read_csv('/content/Constitution Of India.csv')

df.head()

df2 = pd.read_csv('/content/Index.csv', encoding='ISO-8859-1')

df2.head()

# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

df['embeddings'] = df['Articles'].apply(lambda x: model.encode(x, convert_to_tensor=True))

sample1 = "Two siblings inherit an ancestral property after the passing of their parents. While one sibling wishes to sell the property and divide the proceeds, the other wants to retain ownership and preserve the ancestral home. The dispute becomes complicated when the sibling wishing to keep the property refuses to cooperate with any efforts to partition or sell it, citing emotional attachment and family tradition. Legal action becomes necessary to resolve the conflict. The siblings may approach a civil court to either partition the property or settle the dispute under the Hindu Succession Act, 1956, which governs the division of ancestral property among heirs. Additionally, the Indian Partition Act, 1893 may be invoked to determine the division of jointly held property when the co-owners cannot agree on its usage or sale."

sample_embedding = model.encode(sample1, convert_to_tensor=True)

sample_embedding

df['similarity'] = df['embeddings'].apply(lambda x: util.cos_sim(sample_embedding, x).item())

df_sorted = df.sort_values(by='similarity', ascending=False)

top_5 = df_sorted.head(5)
for idx, row in top_5.iterrows():
    print(f"Article Index: {idx}, Similarity Score: {row['similarity']}")
    print(f"Article: {row['Articles']}\n")

# Select the most similar article
most_similar_row = df_sorted.iloc[0]  # Get the top similar article
most_similar_article = most_similar_row['Articles']

from transformers import pipeline

# Load the generative model, for example, T5 or GPT-2
# For T5, specify the model checkpoint
model_name = "t5-small"  # You can also use "gpt2" for GPT-2
generator = pipeline("text2text-generation", model=model_name)  # Use "text-generation" for GPT-2

# Prepare the input text for the generative model
input_text = f"Situation: {sample1}\n\nMost Similar Article: {most_similar_article}\n\nExplain the law and how it affects the situation in simple terms."

# Generate the explanation
explanation = generator(input_text, max_length=150, num_return_sequences=1)

# Print the generated explanation
print("Simplified Explanation:")
print(explanation[0]['generated_text'])  # For T5
# print(explanation[0])  # For GPT-2, you may just want to access the text directly


from transformers import pipeline

# Load the generative model (e.g., T5)
model_name = "t5-small"  # or "gpt2" for GPT-2
generator = pipeline("text2text-generation", model=model_name)

# Prepare the input text for the generative model
input_text = (
    f"Given the following situation, explain the relevant law and its implications in simple terms.\n\n"
    f"Situation: {sample1}\n\n"
    f"Most Similar Article: {most_similar_article}\n\n"
    "Please provide a concise explanation of how the law applies to this situation."
)

# Generate the explanation
explanation = generator(
    input_text,
    max_length=200,  # Adjust for desired length
    num_return_sequences=1,
    temperature=0.7,  # Adjust for creativity
    top_k=50  # Limit vocabulary tokens
)

# Print the generated explanation
print("Simplified Explanation:")
print(explanation[0]['generated_text'])


from transformers import pipeline, AutoTokenizer

# Load the generative model (e.g., T5)
model_name = "t5-small"  # or "gpt2" for GPT-2
generator = pipeline("text2text-generation", model=model_name, device=0)  # Use GPU

# Prepare the input text for the generative model
input_text = (
    f"Given the following situation, explain the relevant law and its implications in simple terms.\n\n"
    f"Situation: {sample1}\n\n"
    f"Most Similar Article: {most_similar_article}\n\n"
    "Please provide a concise explanation of how the law applies to this situation and how it affects it."
)

# Load tokenizer to check the input length
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_tokens = tokenizer(input_text, return_tensors="pt")

# Ensure the token length is within the limit
if len(input_tokens['input_ids'][0]) > 512:
    input_text = input_text[:512]  # Truncate or modify to fit within 512 tokens

# Generate the explanation
explanation = generator(
    input_text,
    max_length=200,  # Adjust for desired length
    num_return_sequences=1,
    temperature=0.7,  # Adjust for creativity
    top_k=50,  # Limit vocabulary tokens
    do_sample=True  # Enable sampling
)

# Print the generated explanation
print("Simplified Explanation:")
print(explanation[0]['generated_text'])


from transformers import pipeline, AutoTokenizer

# Load the generative model (e.g., T5)
model_name = "t5-small"  # or "gpt2" for GPT-2
generator = pipeline("text2text-generation", model=model_name, device=0)  # Use GPU

# Prepare the input text for generating a simplified version of the article
simplified_article_input = (
    f"Given the following article, please provide a simplified version:\n\n"
    f"Article: {most_similar_article}\n\n"
)

# Generate the simplified article
simplified_article_output = generator(
    simplified_article_input,
    max_length=150,  # Adjust for desired length
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    do_sample=True
)

# Prepare the input text for generating how the law affects the situation
impact_on_situation_input = (
    f"Given the following situation, explain how the relevant law affects it in simple terms.\n\n"
    f"Situation: {sample1}\n\n"
    f"Most Similar Article: {most_similar_article}\n\n"
    "Please explain how the law applies to this situation."
)

# Generate the explanation for how the law affects the situation
impact_on_situation_output = generator(
    impact_on_situation_input,
    max_length=250,  # Adjust for desired length
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    do_sample=True
)

# Print the generated outputs
print("Simplified Article:")
print(simplified_article_output[0]['generated_text'])

print("\nImpact on Situation:")
print(impact_on_situation_output[0]['generated_text'])

def Feature_1_RAG(text):
    encoded_text = model.encode(text, convert_to_tensor=True)
    df['similarity'] = df['embeddings'].apply(lambda x: util.cos_sim(encoded_text, x).item())
    df_sorted = df.sort_values(by='similarity', ascending=False)
    most_similar_row = df_sorted
    most_similar_article = most_similar_row['Articles']
    input_text = f"Situation: {sample1}\n\nMost Similar Article: {most_similar_article}\n\nExplain the law and how it affects the situation in simple terms."
    explanation = generator(input_text, max_length=150, num_return_sequences=1)
    return explanation

