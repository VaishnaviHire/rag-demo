# rag-demo
This guide will give a walkthrough on deploying a RAG agent using Ramalama and LLama Stack. The Llama Stack provides a framework 
for building AI agents, and ramalama simplifies running AI models in containers.

## Prerequisites

1. [RamaLama](https://github.com/containers/ramalama?tab=readme-ov-file#install)
2. Podman 
3. Podman Desktop
4. Python 3.10+
5. pip (latest version)

Verify Installations

```
podman --version
python3 --version
pip --version
ramalama version
```

## Create custom network

Create a Podman network to enable communication between containers:
```commandline
podman network create llama-network
```

## Start the Model Server with RamaLama 
Run the ramalama container in server mode to host the model:

1. Ensure you define the container engine for ramalama 
```commandline
export RAMALAMA_CONTAINER_ENGINE=podman
```

2. Run model server 
```commandline
 ramalama serve --network llama-network llama3.2:3b-instruct-fp16 -p 1143
```

## Run Llama Stack Server

1. Setup environment variables
```commandline
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```

2. Run llama-stack server with podman

a) Pull the distribution image
```commandline
podman pull docker.io/llamastack/distribution-ollama:0.1.5
```

b) Make local directory to mount
```commandline
mkdir -p ~/.llama
```

c) Run the server

```commandline
podman run --privileged -it \
  --network host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434 \
  docker.io/llamastack/distribution-ollama:0.1.5 \
  --port $LLAMA_STACK_PORT
```

Currently this fails as we don't yet have support for LamaRama https://github.com/meta-llama/llama-stack/pull/1564 