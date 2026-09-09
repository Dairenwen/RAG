# Project1 deployment

Build the image from the repository root:

```bash
docker build -f Project1/deploy/Dockerfile -t medical-rag .
```

Run it with the model service and API key configured in the environment:

```bash
docker run --rm -p 8000:8000 \
  -e DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
  medical-rag
```

Open `http://localhost:8000` for the frontend. The API health check is available at `http://localhost:8000/api/health`.

The `medical.db` Milvus Lite database is created under `Project1` when the first chat request initializes the collection. The collection must contain indexed rows before retrieval can return medical context.
