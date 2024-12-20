import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import ServerlessSpec, Pinecone
import google.generativeai as genai
import streamlit as st

class Chatbot:
    def __init__(self, pdf_path, pinecone_index_name):

        # Load environment variables
        load_dotenv()

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENV")
        self.pdf_path = pdf_path
        self.pinecone_index_name = pinecone_index_name

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Initialize SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Check if the index exists; create if not
        index_exists = self.check_index_exists(pinecone_index_name)
        if not index_exists:
            print(f"Index '{pinecone_index_name}' does not exist. Creating it...")
            self.create_index(pinecone_index_name)
            self.index = self.pc.Index(self.pinecone_index_name)
            self.create_and_store_embeddings()
        else:
            print(f"Index '{pinecone_index_name}' already exists.")
            self.index = self.pc.Index(pinecone_index_name)

        # Initialize Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.genai_model = genai.GenerativeModel("gemini-1.5-flash")

    def check_index_exists(self, index_name):
        """Checks if the specified Pinecone index exists."""
        indexes = self.pc.list_indexes()
        for index in indexes:
            if index["name"] == index_name:
                return True
        return False
    
    def create_index(self, index_name):
        """Creates a new Pinecone index with serverless specification."""
        self.pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
        print(f"Index '{index_name}' created successfully.")
    
    def process_pdf(self):
        """Reads the PDF and splits text into chunks."""
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return text_splitter.split_text(text)

    def create_and_store_embeddings(self):
        """Processes the PDF, generates embeddings, and stores them in Pinecone."""
        text_chunks = self.process_pdf()
        chunk_embeddings = self.embedding_model.encode(text_chunks)

        vectors = [
            {
                "id": f"chunk-{i}",
                "values": embedding.tolist(),
                "metadata": {"text": text_chunks[i]}
            }
            for i, embedding in enumerate(chunk_embeddings)
        ]

        for vector in vectors:
            self.index.upsert(
                vectors=[
                    {
                        "id": vector["id"],
                        "values": vector["values"],
                        "metadata": vector["metadata"]
                    }
                ]
            )

    def retrieve_context(self, question):
        """Retrieves the most relevant chunks from Pinecone for a given question."""
        question_embedding = self.embedding_model.encode(question)

        results = self.index.query(
            vector=question_embedding.tolist(),
            top_k=10,
            include_metadata=True
        )

        return " ".join([match["metadata"]["text"] for match in results["matches"]])

    def generate_answer(self, context, question):
        """Generates an answer using Gemini API based on the context and question."""
        prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {question}"
        response = self.genai_model.generate_content(prompt)
        return response.text

# Initialize the chatbot once
@st.cache_resource
def init_chatbot():
    pdf_path = "budget_speech.pdf"  # Path to your PDF
    pinecone_index_name = pdf_path.replace(" ", "_").replace("-", "_").replace(".","_").split("_")[0].lower()  # Pinecone index name
    return Chatbot(pdf_path, pinecone_index_name)

# Initialize the chatbot
chatbot = init_chatbot()

# Streamlit interface
st.title("PDF Chatbot with Gemini AI")
st.write("Ask questions about the document!")

# User input for questions
question = st.text_input("Enter your question:")

if question:
    # Retrieve context and generate answer
    with st.spinner("Retrieving answer..."):
        context = chatbot.retrieve_context(question)
        answer = chatbot.generate_answer(context, question)
    st.write("### Answer:")
    st.write(answer)

    # Optionally display the retrieved context
    with st.expander("Show retrieved context"):
        st.write(context)
