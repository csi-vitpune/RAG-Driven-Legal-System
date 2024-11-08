# Install required libraries
# !pip install PyPDF2 sentence-transformers chromadb transformers torch

# Load pdf
import PyPDF2
from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
print(f"Uploaded file: {pdf_path} ")

# Function to validate and get basic information about the PDF
def validate_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    num_pages = len(reader.pages)

    for page_number in range(num_pages):
        page = reader.pages[page_number]
        text = page.extract_text()
        print(f"Page {page_number + 1}: {len(text)} characters")

validate_pdf(pdf_path)

# PDF to chunks
import spacy

# Load English language model for sentence segmentation
nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

# Function to split PDF into sentences and then into chunks
def pdf_to_chunks(pdf_path, chunk_size=10):
    reader = PyPDF2.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_info = []

    for page_number in range(num_pages):
        page = reader.pages[page_number]
        text = page.extract_text()

        # Split text into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        # Create chunks of sentences
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        # Store each page's chunks in a dictionary
        pdf_info.append({
            'page': page_number + 1,
            'chunks': chunks
        })

    return pdf_info

# Get chunks from the PDF
chunks_info = pdf_to_chunks(pdf_path)


# Chunks to Embedding
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

# Convert chunks into embeddings
def create_embeddings(chunks_info):
    all_embeddings = []
    for page_data in chunks_info:
        for chunk in page_data['chunks']:
            embedding = embedding_model.encode(chunk, convert_to_tensor=True)
            all_embeddings.append((page_data['page'], chunk, embedding))
    return all_embeddings

# Create embeddings from the chunks
embeddings = create_embeddings(chunks_info)

# Storing chunks in ChromaDB
import chromadb

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection to store PDF embeddings
collection = client.create_collection(name="pdf_embeddings")

# Store the embeddings with IDs and chunks
for idx, (page, chunk, embedding) in enumerate(embeddings):
    collection.add(
        documents=[chunk],
        embeddings=[embedding.cpu().numpy()],
        ids=[f"doc_{page}_{idx}"]
    )

# Querying ChromaDB
def query_chromadb(query, top_k=1):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

query = "Explain the concept of machine learning."
results = query_chromadb(query)
text = results['ids'][0][0]

def RAG_feature_5(input_text):
    results = query_chromadb(query)
    text = results['ids'][0][0]
    return text