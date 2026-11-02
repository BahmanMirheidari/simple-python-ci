# FastAPI Calculator and RAG Assistant

A Python-based REST API demonstrating software engineering practices, automated testing, Continuous Integration and Continuous Deployment (CI/CD), and Retrieval-Augmented Generation (RAG).

The project combines a basic calculator API with a document-based question-answering system powered by semantic search, FAISS, Sentence Transformers and Ollama.

## 1. Project Features

### Calculator API
- Addition and subtraction operations.
- REST API implementation using FastAPI.
- Unit tests using pytest.
- API integration tests using HTTPX.

### RAG Assistant
- Document ingestion and text extraction.
- Text chunking with configurable overlap.
- Semantic embeddings using Sentence Transformers.
- Vector similarity search using FAISS.
- LLM-based answer generation using Ollama.
- Source attribution for retrieved documents.
- Automated testing using mocked embedding models and LLM responses.

### Software Engineering
- Modular Python application architecture.
- Unit and integration testing.
- Git version control and feature branches.
- Automated CI using GitHub Actions.
- Automated deployment following successful tests (CD).
- Optional Docker containerisation.

## 2. Architecture

```text
fastapi-project/
│
├── calculator.py
├── test_calculator.py
├── main.py
├── test_api.py
│
├── rag/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── generator.py
│
├── documents/
│   └── sample.txt
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   └── test_generator.py
│
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend API | FastAPI |
| API server | Uvicorn |
| Testing | Pytest, HTTPX |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | FAISS |
| LLM | Ollama / Qwen |
| Version control | Git / GitHub |
| CI/CD | GitHub Actions |

## 4. Local Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Linux / macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Install and configure Ollama

Install Ollama from:

https://ollama.com/

Download the selected model:

```bash
ollama pull qwen3:8b
```

Start Ollama:

```bash
ollama serve
```

The default API endpoint is:

```text
http://localhost:11434
```

Ensure the model name in `rag/generator.py` matches the installed model.

## 5. Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to test the calculator and RAG endpoints.

## 6. Running Tests

Run all tests:

```bash
pytest -v
```

Run tests with coverage:

```bash
pip install pytest-cov

pytest --cov=. --cov-report=term-missing
```

Run individual test modules:

```bash
pytest -v test_calculator.py

pytest -v test_api.py

pytest -v tests/test_ingestion.py

pytest -v tests/test_retriever.py

pytest -v tests/test_generator.py
```

RAG unit tests should use mocked embedding models and HTTP responses wherever possible. This avoids downloading large models or requiring a running Ollama server during CI.

Real-model integration tests can be run separately.

## 7. Continuous Integration with GitHub Actions

The project uses GitHub Actions to automatically run tests whenever changes are pushed to GitHub or a pull request is created.

### Step 1: Create the workflow

Create:

```text
.github/workflows/ci.yml
```

Example workflow:

```yaml
name: Python CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: pytest -v
```

The workflow performs the following tasks:

1. Checks out the latest repository code.
2. Sets up Python.
3. Installs project dependencies.
4. Runs the automated test suite.
5. Reports success or failure in GitHub Actions.

Note: Pin action versions and dependencies appropriately when preparing a production deployment.

### Step 2: Push changes to GitHub

Check the current Git status:

```bash
git status
```

Add and commit changes:

```bash
git add .

git commit -m "Add RAG functionality and automated tests"
```

Push changes:

```bash
git push origin main
```

### Step 3: Check GitHub Actions

1. Open your GitHub repository.
2. Select the Actions tab.
3. Open the Python CI workflow.
4. Inspect the test job.
5. Check whether all tests passed.

A successful workflow indicates that the automated tests have passed.

A failed workflow provides logs to help identify the problem.

## 8. Recommended Git Development Workflow

For new features, use feature branches rather than committing directly to the main branch.

Example: adding RAG functionality.

### Step 1: Create a feature branch

```bash
git checkout -b feature/rag-assistant
```

### Step 2: Implement the feature

Add the RAG modules and their tests.

### Step 3: Run local tests

```bash
pytest -v
```

### Step 4: Commit changes

```bash
git add .

git commit -m "Implement RAG document retrieval"
```

### Step 5: Push the feature branch

```bash
git push -u origin feature/rag-assistant
```

### Step 6: Create a Pull Request

On GitHub:

1. Open the repository.
2. Select Compare & pull request.
3. Set the target branch to `main`.
4. Review the changes.
5. Submit the pull request.

GitHub Actions automatically runs the CI workflow against the pull request.

### Step 7: Merge

Merge the pull request only after the tests pass and the changes have been reviewed.

## 9. Continuous Deployment (CD)

Continuous Deployment extends CI by automatically deploying the application after successful testing.

One possible approach is deploying the FastAPI application using Render.

Website:

https://render.com/

Typical deployment configuration:

| Setting | Value |
|---|---|
| Runtime | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Deployment branch | main |

Configure automatic deployments from the main branch.

Important: Ollama is not automatically available on a standard Render deployment. The RAG application needs access to a separately hosted LLM service, or a suitable hosted inference API. The in-memory FAISS index also needs rebuilding or persistent storage.

For this project, the first deployment can focus on the calculator API. RAG deployment can follow once model hosting and vector-index persistence are configured.

## 10. CI/CD Pipeline

The overall development and deployment process is:

```text
Developer
    |
    v
Local development
    |
    v
Run pytest
    |
    v
Git commit
    |
    v
Push feature branch
    |
    v
Create Pull Request
    |
    v
GitHub Actions CI
    |
    v
Automated tests
    |
    +---- Failure --> Fix and push changes
    |
    v
Tests successful
    |
    v
Merge into main
    |
    v
Automatic deployment
    |
    v
Production API
```

## 11. Future Improvements

Potential extensions include:

- Persistent FAISS indexes.
- PDF and DOCX document ingestion.
- Hybrid search combining keyword and semantic retrieval.
- Retrieval evaluation using Precision@K and Recall@K.
- RAG answer evaluation and groundedness checks.
- Docker-based deployment.
- LangChain integration.
- Authentication and API security.
- Structured logging and monitoring.
- LLM-based agentic workflows.

## 12. Learning Objectives

This project demonstrates practical experience with:

- REST API development using Python and FastAPI.
- Modular software architecture.
- Unit and integration testing.
- Git branching and pull-request workflows.
- CI/CD automation.
- Semantic search and vector databases.
- Retrieval-Augmented Generation.
- Local LLM integration.
- Automated software quality checks.

---

**Project status:** Experimental / Learning Project

**License:** MIT (if an MIT licence is added to the repository).