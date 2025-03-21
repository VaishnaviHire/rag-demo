import os
import pickle
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
import time

load_dotenv()

# Configuration
LLAMA_STACK_PORT = os.environ.get("LLAMA_STACK_PORT", "8321")
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "/mnt/models/model.file")
CACHE_DIR = "demo_cache"

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

FILE_PATHS = [
    "documents/kubecon-schedule.rst",
]
TOP_K = 3  # Number of top documents to retrieve


# Function to create an interactive demo section
def demo_section(title, color="blue"):
    cprint("\n" + "=" * 50, color)
    cprint(f" DEMO STEP: {title}", color)
    cprint("=" * 50, color)
    input("Press Enter to continue...")




def create_http_client():
    """Creates an HTTP client for communicating with the Llama Stack server."""
    from llama_stack_client import LlamaStackClient
    endpoint = os.getenv("LLAMA_STACK_ENDPOINT")
    base_url = endpoint if endpoint else f"http://localhost:{LLAMA_STACK_PORT}"
    cprint(f"Connecting to LLM server at: {base_url}", "yellow")
    return LlamaStackClient(base_url=base_url, timeout=1800)



def load_documents():
    documents = [
        Document(
            document_id=f"num-{i}",
            content=Path(file_path).read_text(encoding="utf-8"),
            mime_type="text/plain",
            metadata={"source": file_path}
        )
        for i, file_path in enumerate(FILE_PATHS) if Path(file_path).exists()
    ]

    # Print warnings for missing files
    missing_files = [file for file in FILE_PATHS if not Path(file).exists()]
    if missing_files:
        cprint(f"Warning: {', '.join(missing_files)} not found. Skipping...", "red")

    # Initialize document texts array
    document_texts = [doc.content for doc in documents]
    cprint(f"✓ Processed {len(documents)} documents ready for retrieval", "green")

    return documents, document_texts




def retrieve(query, document_texts, client, k=TOP_K):
    # Prepare evaluation rows for scoring
    eval_rows = []
    scores = []
    for i, doc_text in enumerate(document_texts):
        # Limit text size for demo purposes
        truncated_text = doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text
        eval_rows.append({
            "input_query": query,
            "generated_answer": truncated_text,
            "expected_answer": "",  # Not needed for similarity scoring
        })

    try:
        scoring_params = {
            "basic::subset_of": None,
        }

        cprint("Calling Llama Stack scoring API...", "yellow")
        scoring_response = client.scoring.score(
            input_rows=eval_rows, scoring_functions=scoring_params
        )

        # Extract scores and sort documents
        score_value = scoring_response.results['basic::subset_of'].score_rows[0]['score']
        scores.append((i, score_value))

        # Sort by score (descending)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in sorted_scores[:k]]
        retrieved_docs = [document_texts[idx] for idx in top_indices]

    except Exception as e:
        # Simple fallback if scoring API fails
        retrieved_docs = document_texts[:k]
    return retrieved_docs, sorted_scores



def build_prompt(query, retrieved_docs):
    # Join retrieved documents with separators
    context = "\n\n----\n\n".join(retrieved_docs)
    prompt_template = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"

    return prompt_template


client = create_http_client()
documents, document_texts = load_documents()
rag_agent = Agent(client, model=INFERENCE_MODEL,
                  instructions="You should answer questions based on the provided context.")
session_id = rag_agent.create_session("rag-demo-session")


# Load response cache
response_cache = {}
response_cache_file = f"{CACHE_DIR}/response_cache.pkl"
if os.path.exists(response_cache_file):
    with open(response_cache_file, "rb") as f:
        response_cache = pickle.load(f)
    cprint(f"Loaded {len(response_cache)} cached responses", "green")



def get_response(query, use_cache=True):
    if use_cache and query in response_cache:
        return response_cache[query]
    retrieved_docs, top_matches = retrieve(query, document_texts, client)

    full_prompt = build_prompt(query, retrieved_docs)

    response_text = None
    try:
        # Try standard approach first
        response = rag_agent.create_turn(
            messages=[{"role": "user", "content": full_prompt}],
            session_id=session_id,
        )

        # Extract response text from event logger
        cprint("\nRAG response:", "yellow")
        for log in EventLogger().log(response):
            log.print()
            if hasattr(log, 'content') and log.content:
                response_text = log.content
    except Exception as e:
        cprint(f"Error with standard approach: {str(e)}", "red")

    # Cache the response
    if response_text:
        response_cache[query] = response_text
        with open(response_cache_file, "wb") as f:
            pickle.dump(response_cache, f)
        cprint("\n✓ Response cached for future use", "green")

    return response_text




def demo_query(query):
    """Run a query and explain each step."""
    cprint("\n" + "=" * 60, "magenta")
    cprint(f" USER PROMPT: '{query}'", "magenta")
    cprint("=" * 60, "magenta")

    cprint("\n[STEP 1] Document Retrieval", "blue")
    start_time = time.time()
    retrieved_docs, top_matches = retrieve(query, document_texts,client)
    retrieval_time = time.time() - start_time

    cprint(f"✓ Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds", "green")
    cprint("\nTop matches:", "yellow")
    for rank, (idx, score) in enumerate(top_matches):
        doc_preview = document_texts[idx][:50].replace("\n", " ") + "..."
        cprint(f"  #{rank + 1}: Score {score:.2f} - {doc_preview}", "cyan")

    cprint("\n[STEP 2] Prompt Construction", "blue")
    full_prompt = build_prompt(query, retrieved_docs)
    cprint(f"✓ Created prompt with {len(full_prompt)} characters", "green")
    cprint("\nPrompt structure:", "yellow")
    cprint("  1. Context from retrieved documents", "cyan")
    cprint("  2. User query: " + query, "cyan")
    cprint("  3. Answer instruction", "cyan")

    cprint("\n[STEP 3] Generated Response from model", "blue")
    if query in response_cache:
        cprint("✓ Using cached response (instant!)", "green")
        response_text = response_cache[query]
    else:
        cprint("Sending prompt LLM model...", "yellow")
        start_time = time.time()
        try:
            response = rag_agent.create_turn(
                messages=[{"role": "user", "content": full_prompt}],
                session_id=session_id,
            )
            # Extract response text from event logger
            for log in EventLogger().log(response):
                log.print()
                if hasattr(log, 'content') and log.content:
                    response_text = log.content

            gen_time = time.time() - start_time
            cprint(f"✓ Generated response in {gen_time:.2f} seconds", "green")
        except Exception as e:
            cprint(f"Error: {str(e)}", "red")
            response_text = f"Error: {str(e)}"

    # Display final response
    cprint("\n[FINAL RESPONSE]", "blue")
    cprint(response_text, "white")

    # Return for use in IPython
    return response_text



def initialize_demo():
    cprint("\n" + "=" * 60, "green")
    cprint(" KUBECON AGENTIC DEMO - INITIALIZED ", "green")
    cprint("=" * 60, "green")
    cprint("\nRunning a sample query to demonstrate the system...", "white")

    sample_query = "When is the demo on AI agents?"
    demo_query(sample_query)
    cprint("\n\nDemo initialized successfully! Use rag_agent.get_response('Your query'') to try your own queries.","yellow")


# Make functions available to IPython
if __name__ == "__main__":
    initialize_demo()
else:
    initialize_demo()