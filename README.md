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
 ramalama --image=quay.io/bluesman/vllm-cpu-env:latest --runtime vllm serve meta-llama/Llama-3.2-3B-Instruct
```

## Run Llama Stack Server

1. Setup environment variables
```commandline
export INFERENCE_MODEL="/mnt/models/model.file"
export LLAMA_STACK_PORT=8321
```

2. Run llama-stack server with podman

```commandline
podman run \
  -it --network=host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-remote-vllm \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_MAX_TOKENS=200 \
  --env VLLM_API_TOKEN=fake \
  --env VLLM_URL=http://localhost:8080/v1 
```

3. Test using llama-stack-client

```commandline
llama-stack-client --endpoint http://localhost:8321 inference chat-completion --message "hello, what model are you?"

```

### Rag agent implementation

```commandline
python3 rag-agent.py
```
