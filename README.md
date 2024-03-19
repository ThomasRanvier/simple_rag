# Simple Implementation of RAG using Mistral 7B

## Introduction

This repository demonstrates a simple implementation of Retrieval-Augmented Generation (RAG) utilizing the open-source language model, Mistral 7B, paired with an open-source embedding model specifically designed for the French language, Sentence Camembert Large.

Inspired by [ask-multiple-pdfs](https://github.com/alejandro-ao/ask-multiple-pdfs/blob/main/app.py).

This implementation focuses on providing enriched answers by retrieving relevant information from a set of documents in French.

## Features

- Utilization of Mistral 7B, an open-source LLM, for generating responses.
- Integration of Sentence Camembert Large for French embeddings for semantic similarity retrieval on French documents.
- Dockerized backend and frontend services for easy setup and deployment.

## Installation

### Prerequisites

- Docker and Docker Compose installed on your machine.
- A GPU.

### Setup

#### Option 1: Simultaneously start both services

1. **Clone the Repository**

```bash
git clone https://github.com/ThomasRanvier/simple_rag.git
cd simple_rag
```

2. **Start the Services**

Use Docker Compose to start both the backend and frontend services together.

```bash
docker compose up
```

This command reads the docker-compose.yml file at the root of the repository, which is configured to start both services as part of a single Docker network.

Once the services are up, open [http://localhost:8081](http://localhost:8081) in your web browser to access the application.

#### Option 2: Manually start each service

1. **Clone the Repository**

```bash
git clone https://github.com/ThomasRanvier/simple_rag.git
cd simple_rag
```

2. **Running the Backend API**

Navigate to the backend directory and start the Docker container.

```bash
cd backend
docker compose up
```

3. **Running the Frontend API**

Make sure the backend service is up and running before starting the frontend to ensure the shared network is up.

```bash
cd frontend
docker compose up
```

After successfully starting the services, open [http://localhost:8081](http://localhost:8081) in your web browser to interact with the application.

## License

[MIT](https://choosealicense.com/licenses/mit/)