import os
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
from rank_bm25 import BM25Okapi

load_dotenv()

# Configuration
LLAMA_STACK_PORT = os.environ.get("LLAMA_STACK_PORT", "8321")
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "/mnt/models/model.file")

FILE_PATHS = [
    "documents/capital.rst",
]
TOP_K = 3  # Number of top documents to retrieve
MAX_TOKENS_IN_CONTEXT = 4096  # Context size limit (for reference)

def create_http_client():
    """Creates an HTTP client for communicating with the Llama Stack server."""
    from llama_stack_client import LlamaStackClient
    endpoint = os.getenv("LLAMA_STACK_ENDPOINT")
    base_url = endpoint if endpoint else f"http://localhost:{LLAMA_STACK_PORT}"
    return LlamaStackClient(base_url=base_url,timeout=1200)

client = create_http_client()

# Load documents
documents = [
    Document(
        document_id=f"num-{i}",
        content=Path(file_path).read_text(encoding="utf-8"),
        mime_type="text/plain",
        metadata={}
    )
    for i, file_path in enumerate(FILE_PATHS) if Path(file_path).exists()
]

# Print warnings for missing files
missing_files = [file for file in FILE_PATHS if not Path(file).exists()]
if missing_files:
    print(f"Warning: {', '.join(missing_files)} not found. Skipping...")

# Initialize BM25 retrieval
document_texts = [doc.content for doc in documents]
tokenized_docs = [doc.split() for doc in document_texts]  # Simple tokenization
bm25 = BM25Okapi(tokenized_docs)

# Define retrieval function
def retrieve(query, k=TOP_K):
    """Retrieve top-k documents using BM25."""
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [document_texts[i] for i in top_k_indices]

# Define prompt construction function
def build_prompt(query, retrieved_docs):
    """Build a prompt with context and query."""
    context = "\n\n".join(retrieved_docs)
    return f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"


# Create agent and session
rag_agent = Agent(client,model=INFERENCE_MODEL, instructions="You should always use the RAG tool to answer questions.")
session_id = rag_agent.create_session("test-session")

# User prompts
user_prompts = [
    "What is the capital of France?",
]

# Run the agent loop
for prompt in user_prompts:
    cprint(f"User> {prompt}", "green")
    # Retrieve relevant documents
    retrieved_docs = retrieve(prompt)
    # Build prompt with context
    full_prompt = build_prompt(prompt, retrieved_docs)
    # Create turn with enriched prompt
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": full_prompt}],
        session_id=session_id,
    )
    for log in EventLogger().log(response):
        log.print()