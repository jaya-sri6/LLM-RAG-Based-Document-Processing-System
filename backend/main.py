from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import openai
import os
from typing import List

# --- App Initialization ---
app = FastAPI(
    title="Insurance Policy Q&A System",
    description="An LLM-powered system to answer questions about insurance policies.",
)

# --- In-Memory Data Store ---
# For a hackathon, an in-memory store is simple and fast.
# For production, you'd use a database and a persistent vector store.
document_store = {
    "filename": "",
    "text": "",
    "chunks": [],
    "embeddings": np.array([]),
}

# --- OpenAI API Client ---
# It's good practice to initialize the client once.
# The API key is read from the OPENAI_API_KEY environment variable.
try:
    openai_client = openai.OpenAI()
    # A simple check to see if the key is present.
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set. API calls will fail.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None


@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "ok", "message": "Welcome to the Insurance Q&A System!"}


@app.post("/upload/", summary="Upload and Process a Document")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (.pdf), parses it, chunks it, and stores it in memory.
    This prepares the document for the embedding step.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported for now.")

    temp_dir = "/tmp/insurancedocs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # 1. Parse the PDF using pdfplumber
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")

        # 2. Chunk the text using LangChain's splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # The size of each chunk in characters
            chunk_overlap=50,  # The overlap between consecutive chunks
            length_function=len,
        )
        chunk_texts = text_splitter.split_text(full_text)

        # 3. Store the processed data in our in-memory store
        document_store["filename"] = file.filename
        document_store["text"] = full_text
        document_store["chunks"] = chunk_texts
        document_store["embeddings"] = np.array([])  # Reset embeddings on new upload

        return {
            "message": f"Successfully uploaded and processed '{file.filename}'",
            "filename": file.filename,
            "num_chunks": len(chunk_texts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/embed/", summary="Create Embeddings for the Uploaded Document")
def embed_document():
    """
    Generates vector embeddings for the currently loaded document's chunks
    using OpenAI's API and stores them in the in-memory vector store.
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client is not initialized. Is the OPENAI_API_KEY set?")

    if not document_store["chunks"]:
        raise HTTPException(status_code=404, detail="No document has been uploaded. Please use the /upload endpoint first.")

    if document_store.get("embeddings") is not None and document_store["embeddings"].any():
        return {"message": "Embeddings have already been created for the current document."}

    try:
        # 4. Generate embeddings using OpenAI's text-embedding-3-small model
        response = openai_client.embeddings.create(
            input=document_store["chunks"],
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]

        # Store as a NumPy array for efficient similarity calculations
        document_store["embeddings"] = np.array(embeddings)

        return {
            "message": f"Successfully created {len(embeddings)} embeddings for '{document_store['filename']}'.",
            "num_embeddings": len(embeddings),
            "embedding_shape": document_store["embeddings"].shape,
        }
    except openai.APIError as e:
        # Handle OpenAI specific API errors gracefully
        error_message = e.body.get("message", str(e)) if e.body else str(e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {error_message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during embedding: {str(e)}")


from pydantic import BaseModel
import json

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# --- Helper Functions ---
def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Handle zero vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

@app.post("/query/", summary="Ask a Question About the Document")
async def query_document(request: QueryRequest):
    """
    Answers a question based on the uploaded document using a RAG pipeline.
    This involves embedding the query, finding relevant document chunks,
    and using a powerful LLM to generate a structured JSON response.
    """
    if document_store["embeddings"].size == 0:
        raise HTTPException(status_code=404, detail="No embeddings found. Please upload and embed a document first.")

    # 1. Embed the user's query
    try:
        response = openai_client.embeddings.create(
            input=[request.query],
            model="text-embedding-3-small"
        )
        query_embedding = np.array(response.data[0].embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {str(e)}")

    # 2. Perform semantic search to find top-k similar chunks
    similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in document_store["embeddings"]]

    # Get the indices of the top-k similarities in descending order
    top_k_indices = np.argsort(similarities)[-request.top_k:][::-1]

    relevant_chunks = [document_store["chunks"][i] for i in top_k_indices]

    # 3. Construct the prompt for the reasoning model
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt_template = f"""
You are an expert insurance claim analyst. Your task is to analyze a user's query based on the provided policy document excerpts and determine if the claim should be approved.

**User Query:** "{request.query}"

**Relevant Policy Clauses:**
---
{context}
---

**Instructions:**
1.  Carefully read the user's query and the relevant policy clauses.
2.  Reason through the clauses to make a decision: "approved", "rejected", or "more_info_needed".
3.  If approved, determine the payout amount if possible from the context. If not, state "As per policy".
4.  Write a clear, concise justification for your decision, referencing the clauses.
5.  Identify the specific clauses that match the query.
6.  You MUST respond with a single, valid JSON object and nothing else. Do not include any text before or after the JSON.
7.  The JSON object must conform to the following schema, including the 'highlights' field which is currently an empty list:
    {{
      "decision": "approved" | "rejected" | "more_info_needed",
      "amount": "string",
      "justification": "string",
      "matched_clauses": [
        {{
          "clause_id": "string (e.g., 'Clause 5.1' or 'N/A' if not applicable)",
          "text": "string",
          "document": "{document_store['filename']}"
        }}
      ],
      "highlights": []
    }}

Now, provide your analysis as a JSON object.
"""

    # 4. Call the Chat Completions API using JSON mode
    try:
        chat_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert insurance claim analyst that provides answers in JSON format."},
                {"role": "user", "content": prompt_template}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        response_content = chat_response.choices[0].message.content
        json_response = json.loads(response_content)

        return json_response

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="The model did not return valid JSON. The raw response was: {response_content}")
    except openai.APIError as e:
        error_message = e.body.get("message", str(e)) if e.body else str(e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error during reasoning: {error_message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during reasoning: {str(e)}")
