# FastAPI Calculator and RAG Assistant

A Python REST API demonstrating backend development, automated testing, Git-based collaboration, CI/CD, and Retrieval-Augmented Generation (RAG).

The project combines a calculator API with a document question-answering system. The RAG pipeline uses Ollama to generate embeddings and LLM responses, with NumPy-based vector similarity search.

**Project repository:** https://github.com/BahmanMirheidari/simple-python-ci

**API documentation:** Open `/docs` on the deployed or locally running application.

## 1. Features

### Calculator API
- Addition, subtraction and division.
- REST API implemented using FastAPI.
- Input validation and error handling.
- Automated unit and API tests.

### RAG Assistant
- Document ingestion and text chunking.
- Text embeddings generated using Ollama.
- NumPy-based cosine similarity retrieval.
- LLM-based answer generation using Qwen.
- Source attribution for retrieved documents.
- Automated tests using mocked models and HTTP responses.

### Software Engineering
- Modular Python architecture.
- Unit and integration testing with pytest.
- Git branching and pull-request workflow.
- Continuous Integration using GitHub Actions.
- Deployment using Render.
- API documentation through FastAPI Swagger UI.

## 2. Project Architecture

```text
simple-python-ci/
│
├── main.py
├── calculator.py
├── test_calculator.py
├── test_api.py
│
├── rag/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── generator.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   └── test_generator.py
│
├── documents/
│   └── sample.txt
│
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

### Main components

| File | Responsibility |
|---|---|
| `main.py` | FastAPI application and API endpoints |
| `calculator.py` | Calculator operations |
| `test_calculator.py` | Calculator unit tests |
| `test_api.py` | API integration tests |
| `rag/ingestion.py` | Document loading and text chunking |
| `rag/embeddings.py` | Embedding generation through Ollama |
| `rag/retriever.py` | Vector normalisation and similarity search |
| `rag/generator.py` | Prompt construction and LLM answer generation |
| `tests/` | Automated RAG component tests |
| `requirements.txt` | Python dependencies |
| `.github/workflows/` | GitHub Actions CI configuration |

## 3. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| API framework | FastAPI |
| API server | Uvicorn |
| Data validation | Pydantic |
| Testing | Pytest, HTTPX |
| Numerical computing | NumPy |
| Embedding service | Ollama |
| Embedding model | `nomic-embed-text` |
| LLM | `qwen3:8b` |
| Version control | Git and GitHub |
| CI | GitHub Actions |
| Deployment | Render |

## 4. Local Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/BahmanMirheidari/simple-python-ci.git

cd simple-python-ci
```

### Step 2: Create a virtual environment

```bash
python3.12 -m venv .venv

source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Install Ollama

Download Ollama from:

https://ollama.com/

Download the embedding and language models:

```bash
ollama pull nomic-embed-text

ollama pull qwen3:8b
```

Start Ollama:

```bash
ollama serve
```

The default Ollama API endpoint is:

```text
http://localhost:11434
```

The application uses:
- `nomic-embed-text` for document and query embeddings.
- `qwen3:8b` for generating answers from retrieved context.

Make sure the model names in `rag/embeddings.py` and `rag/generator.py` match the installed models.

## 5. Running the API Locally

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Swagger UI allows you to inspect endpoints, submit requests and view JSON responses directly in the browser.

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| GET | `/add` | Add two numbers |
| GET | `/subtract` | Subtract two numbers |
| GET | `/divide` | Divide two numbers |
| POST | `/ingest` | Ingest documents and build the retrieval index |
| POST | `/query` | Retrieve relevant content and generate an answer |

### Testing the calculator

Example:

```text
GET /add?a=10&b=5
```

Expected response:

```json
{
  "a": 10,
  "b": 5,
  "result": 15
}
```

### Testing document ingestion

First, open `/docs` and select `POST /ingest`.

Example JSON request:

```json
{
  "documents": [
    {
      "text": "Dementia can affect memory, language, thinking and everyday activities.",
      "source": "dementia_guide.txt"
    },
    {
      "text": "Regular physical activity and social engagement may support general brain health.",
      "source": "brain_health.txt"
    }
  ]
}
```

The endpoint creates chunks, generates embeddings and builds the retrieval index.

A successful response should include the number of documents and chunks processed.

### Testing RAG question answering

After successfully ingesting documents, select `POST /query`.

Example JSON request:

```json
{
  "question": "What are some effects of dementia?",
  "top_k": 3
}
```

The application performs the following steps:

1. Converts the question into an embedding using Ollama.
2. Compares the question embedding with stored document embeddings.
3. Retrieves the most similar text chunks.
4. Constructs a prompt containing the retrieved context.
5. Sends the prompt to Qwen through Ollama.
6. Returns the generated answer and source documents.

The response contains:
- `answer`: generated response.
- `sources`: source documents used.
- `retrieved_chunks`: retrieved text and similarity scores.

Try changing the question and `top_k` value to observe how retrieval affects the response.

**Important:** The current retrieval index is stored in application memory. Ingest documents again after restarting the application.

## 6. Running Automated Tests

Run all tests:

```bash
python -m pytest -v
```

Run individual test modules:

```bash
python -m pytest -v test_calculator.py

python -m pytest -v test_api.py

python -m pytest -v tests/test_ingestion.py

python -m pytest -v tests/test_embeddings.py

python -m pytest -v tests/test_retriever.py

python -m pytest -v tests/test_generator.py
```

The automated tests should use mock embedding models and mocked HTTP responses. This means GitHub Actions does not need a running Ollama server or downloaded LLM models to execute the unit tests.

Real-model integration testing can be performed locally with Ollama.

## 7. Git and GitHub Development Workflow

The project uses feature branches to separate development work from the main branch.

### Step 1: Check the current branch

```bash
git branch
git status
```

### Step 2: Create a feature branch

For example:

```bash
git checkout -b feature/rag-improvements
```

### Step 3: Implement and test changes

Make changes to the source code and run:

```bash
python -m pytest -v
```

### Step 4: Commit changes

```bash
git add .

git commit -m "Improve RAG embedding implementation"
```

### Step 5: Push the feature branch

```bash
git push -u origin feature/rag-improvements
```

For subsequent changes to the same branch:

```bash
git push
```

### Step 6: Create a Pull Request

On GitHub:

1. Open the repository.
2. Select **Pull requests**.
3. Select **New pull request**, or use **Compare & pull request** after pushing a branch.
4. Select the feature branch as the source.
5. Select `main` as the target branch.
6. Review the changes and create the pull request.

### Step 7: Review automated checks

GitHub Actions automatically runs the CI workflow when a pull request is created or updated.

Open the pull request and inspect the checks:

- `Python tests / test (pull_request)`
- `Python tests / test (push)`

If a check fails:

1. Open the failed check.
2. Select the test job.
3. Inspect the error logs.
4. Fix the issue locally.
5. Run pytest again.
6. Commit and push the fix to the same feature branch.

The pull request will update automatically.

### Step 8: Merge the pull request

Once the changes have been reviewed and required checks pass, merge the pull request into `main`.

Update the local main branch afterwards:

```bash
git checkout main

git pull origin main
```

## 8. Continuous Integration with GitHub Actions

The project uses GitHub Actions to automate testing.

The CI workflow performs the following tasks:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs dependencies.
4. Runs pytest.
5. Reports whether the tests pass or fail.

The workflow is triggered by pushes and pull requests according to the configured branches in `.github/workflows/ci.yml`.

This helps identify errors before changes are merged into the main branch.

## 9. Deployment using Render

The FastAPI application can be deployed using Render.

Website:

https://render.com/

### Step 1: Create a Render account

Sign in using your GitHub account and authorise access to the repository.

### Step 2: Create a Web Service

1. Open the Render dashboard.
2. Select **New**.
3. Select **Web Service**.
4. Connect the `simple-python-ci` GitHub repository.
5. Select the deployment branch.

For normal production deployment, use `main` after merging the feature branch.

### Step 3: Configure the service

| Setting | Value |
|---|---|
| Runtime | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |

Create the service and wait for the deployment to complete.

### Step 4: Test the deployed application

Render provides a public URL, for example:

```text
https://YOUR-RENDER-SERVICE.onrender.com
```

Open the API documentation:

```text
https://YOUR-RENDER-SERVICE.onrender.com/docs
```

Test the health endpoint:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
```

You can also test the calculator endpoints through Swagger UI.

### Important: RAG model hosting

The current RAG implementation uses Ollama at:

```text
http://localhost:11434
```

This works when the application and Ollama are running on the same local machine.

A Render service cannot automatically access Ollama running on your Mac. Therefore, the RAG `/ingest` and `/query` endpoints require a separately hosted, reachable embedding and LLM service before they can work on Render.

The application also stores its retrieval index in memory, so persistent storage or re-ingestion is needed when the service restarts.

Until remote model access is configured, treat the Render deployment as a deployment of the API and calculator functionality, not a fully operational hosted RAG service.

## 10. CI/CD Development Pipeline

The intended workflow is:

```text
Local development
       |
       v
Run automated tests
       |
       v
Create feature branch
       |
       v
Commit changes
       |
       v
Push to GitHub
       |
       v
Create Pull Request
       |
       v
GitHub Actions CI
       |
       v
Run automated tests
       |
       +------ Failure
       |          |
       |          v
       |      Fix and push
       |          |
       |          +----> Run checks again
       |
       v
All required checks pass
       |
       v
Merge into main
       |
       v
Render deployment
       |
       v
Test deployed API
```

## 11. Future Improvements

- Persistent vector storage.
- PDF and DOCX ingestion.
- Hybrid keyword and semantic retrieval.
- Retrieval evaluation using Precision@K and Recall@K.
- RAG answer quality and groundedness evaluation.
- Docker containerisation.
- LangChain integration.
- Authentication and API security.
- Structured logging and monitoring.
- Remote LLM and embedding service configuration.
- Agentic AI workflows.

## 12. Learning Outcomes

This project provides practical experience with:

- Python backend development.
- REST API design using FastAPI.
- Document processing and chunking.
- Embedding generation and semantic search.
- Retrieval-Augmented Generation.
- Local LLM integration.
- Unit and API integration testing.
- Git branching and pull-request workflows.
- Continuous Integration using GitHub Actions.
- Cloud application deployment using Render.

---

**Project status:** Experimental learning and portfolio project.

**License:** Add an MIT licence file if distributing the project under the MIT licence.