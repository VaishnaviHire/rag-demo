# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (we'll create this next)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RAG script and documents directory
COPY rag_agent.py .
COPY documents/ ./documents/

# Set environment variables (can be overridden at runtime)
ENV LLAMA_STACK_ENDPOINT="http://localhost:8321"
ENV INFERENCE_MODEL="/mnt/models/model.file"
ENV SERVER_PORT=5001

EXPOSE 5001

ENTRYPOINT ["python", "rag_agent.py"]