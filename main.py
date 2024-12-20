# Import necessary libraries
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Access the variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Step 1: Read the PDF file
reader = PdfReader("budget_speech.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Step 2: Split the extracted text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
text_chunks = text_splitter.split_text(text)

# Step 3: Initialize the SentenceTransformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
# Generate embeddings for each text chunk
chunk_embeddings = model.encode(text_chunks)

# Step 4: Initialize Pinecone and connect to an index
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("budget-speech")

# Prepare data for upserting into Pinecone
vectors = [
    {
        "id": f"chunk-{i}",  # Unique ID for each chunk
        "values": embedding.tolist(),  # Convert NumPy array to a list
        "metadata": {"text": text_chunks[i]}  # Store the original text as metadata
    }
    for i, embedding in enumerate(chunk_embeddings)
]

# Upsert (insert/update) each chunk embedding into Pinecone
for vector in vectors:
    index.upsert(
        vectors=[
            {
                "id": vector["id"],
                "values": vector["values"],
                "metadata": vector["metadata"]
            }
        ]
    )

# Step 5: Configure Gemini API
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Step 6: Enter a loop for endless questions
while True:
    # Prompt the user for a question
    question = input("Enter your question (type 'exit' to quit): ")
    
    # Check if the user wants to exit
    if question.lower() == 'exit':
        print("Exiting the chatbot. Goodbye!")
        break
    
    # Generate embedding for the user's question
    question_embedding = model.encode(question)

    # Query Pinecone for the most relevant chunks
    results = index.query(
        vector=question_embedding.tolist(),
        top_k=10,
        include_metadata=True
    )

    # Combine the retrieved chunks into a single context
    retrieved_texts = " ".join([match["metadata"]["text"] for match in results["matches"]])
    # print(retrieved_texts)

    # Prepare the prompt for the generative AI model
    prompt = f"Based on the following context, answer the question:\n\n{retrieved_texts}\n\nQuestion: {question}"

    # Generate the answer using Gemini API
    response = gemini_model.generate_content(prompt)

    # Print the answer
    print(f"Answer: {response.text}\n")