# Psychiatric Symptom Pattern Analyzer

## Overview

**Psychiatric Symptom Pattern Analyzer** is an intelligent medical consultation application that uses artificial intelligence and evidence-based retrieval to analyze psychiatric symptom descriptions and deliver clinically relevant insights. The application combines natural language processing, retrieval-augmented generation (RAG), and conversational memory to produce structured psychiatric assessment output.

![Application Screenshot](image.png)

## Key Features

- **Symptom Analysis**: Converts natural language psychiatric symptom descriptions into structured clinical output
- **Conversational Memory**: Maintains conversation history and thread context for each user session
- **Medical RAG Integration**: Retrieves evidence-based medical context from a curated vector store and external sources
- **Clinical Reasoning**: Uses multi-stage psychiatric prompt engineering for robust synthesis
- **User Sessions**: Tracks individual user sessions with cookies and thread persistence
- **Streaming Responses**: Supports real-time analysis with server-sent event updates

## Application Architecture

### Frontend
- **Framework**: Angular 21.2.0
- **Styling**: SCSS with Bootstrap 5.3.3
- **UI Features**:
  - Interactive symptom input interface
  - Conversation thread management
  - Markdown rendering of medical analysis
  - Security-conscious HTML sanitization with DOMPurify

### Backend
- **Framework**: Flask with CORS support
- **Port**: 5000
- **Core Capabilities**:
  - RESTful API endpoints for symptom analysis
  - User session and cookie-based authentication
  - Conversation thread management and persistence
  - Real-time streaming responses via SSE
  - FastAPI alternative available via `backend/run_fast.py`

### AI & ML Stack
- **LLM**: Ollama with `qwen2.5:7b` for psychiatric reasoning
- **NLP**:
  - Transformers library for semantic analysis
  - PyTorch for model execution
  - Clinical NER for symptom enrichment
- **RAG System**:
  - ChromaDB for vector database and embeddings
  - Medical corpus ingestion pipeline
  - Semantic retrieval of medical evidence
- **Medical Data Processing**:
  - PyMuPDF for PDF processing
  - BioPython for PubMed/Entrez integration
  - HuggingFace datasets support

### Database & Storage
- **Primary**: MongoDB for conversation history and analysis transcripts
- **Secondary**: PostgreSQL for user, thread, and memory metadata
- **Vector DB**: ChromaDB for semantic embeddings and medical retrieval

## Technology Stack

### Backend Dependencies
```
flask              - Web framework
pymongo            - MongoDB driver
psycopg2-binary    - PostgreSQL driver
requests           - HTTP client
ollama             - Ollama client
transformers       - NLP models
torch              - Deep learning framework
torchvision        - Model utilities
python-dotenv      - Environment configuration
chromadb           - Vector database
pymupdf            - PDF processing
datasets           - HuggingFace datasets
biopython          - NCBI / PubMed integration
flask-cors         - Cross-origin request handling
```

### Frontend Dependencies
```
@angular/*         - Angular framework (v21.2.0)
bootstrap          - UI framework (v5.3.3)
marked             - Markdown parser (v15.0.0)
dompurify          - HTML sanitization (v3.2.0)
rxjs               - Reactive programming (v7.8.0)
```

## How It Works

### Clinical Analysis Pipeline

1. **User Input**: Patient or clinician enters psychiatric symptoms in the Angular UI
2. **Session Management**: Backend reads or creates a `user_id` cookie to maintain context
3. **Clinical Reformulation**: The analyzer reforms symptom text into structured psychiatric variants
4. **Medical Retrieval**: RAG retrieves relevant evidence from ChromaDB and external sources
5. **Query Verification**: PubMed and ICD-11 queries validate clinical relevance
6. **Analysis Output**: The backend synthesizes a structured psychiatric response
7. **Memory Persistence**: Conversation history and thread memory are stored for future use

### Data Flow

```
User Input
    ↓
Flask Backend (/analyze endpoint)
    ↓
Query Preprocessor → Clinical Reformulation
    ↓
Medical RAG Retriever (ChromaDB)
    ↓
Medical Query Verifier
    ↓
Ollama LLM (Qwen2.5:7b)
    ↓
Conversation Memory (MongoDB/PostgreSQL)
    ↓
JSON Response Stream → Angular Frontend
    ↓
UI Rendering & Display
```

## Project Structure

```
.
├── backend/                    # Python backend and AI engine
│   ├── engine/
│   │   ├── analyzer.py          # Core psychiatric analysis pipeline
│   │   ├── main.py              # Flask app and API routes
│   │   ├── main_fastapi.py      # FastAPI alternative app
│   │   ├── rag_ingestion.py     # Medical corpus ingestion routines
│   │   ├── rag_retriever.py     # ChromaDB retrieval logic
│   │   ├── verifier.py          # PubMed / ICD-11 evidence validation
│   │   ├── memory.py            # Conversation and thread memory
│   │   ├── database.py          # MongoDB/PostgreSQL connections and schema
│   │   ├── data_model.py        # Data models and schema definitions
│   │   └── query_preprocess.py  # Clinical NLP preprocessing
│   ├── medcpt_db/               # ChromaDB persistent vector store
│   ├── run.py                   # Flask entrypoint
│   ├── run_fast.py              # FastAPI entrypoint
│   ├── constants.py             # Ollama connection constants
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Backend container build
├── frontend/                   # Angular frontend application
│   ├── src/
│   │   ├── app/
│   │   │   ├── main/            # Main component
│   │   │   ├── mydoc/           # Document/analysis view module
│   │   │   └── services/        # API client service
│   │   ├── index.html
│   │   ├── main.ts
│   │   └── styles.scss
│   ├── angular.json
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.yml         # Container orchestration
└── README.md                  # Project documentation
```

## API Endpoints

### User Management
- `GET /me` - Get or create a `user_id` cookie
- `GET /threads` - Retrieve all conversation threads for the user
- `GET /thread/{thread_id}` - Retrieve the history for a specific thread

### Analysis
- `POST /analyze` - Analyze symptoms with streaming response updates
- `POST /fetch-analysis` - Fetch final stored analysis for a thread
- `GET /reset-rag` - Re-ingest the medical RAG corpus

## Running the Application

### Using Docker Compose

```bash
docker compose up --build
```

Open the frontend at:

```text
http://localhost:4200
```

The backend is exposed at:

```text
http://localhost:5000
```

### Manual Setup

**Backend**:

```bash
cd backend
python -m pip install -r requirements.txt
python run.py
```

**FastAPI alternative**:

```bash
python run_fast.py
```

**Frontend**:

```bash
cd frontend
npm install
npm start
```

## Configuration

Create a `.env` file in `backend/` with values such as:

```
MONGO_URI=mongodb://localhost:27017/pspa
POSTGRES_URI=postgresql://user:pass@localhost:5432/pspa
OLLAMA_BASE_URL=http://host.docker.internal:11434
FLASK_ENV=development
FLASK_DEBUG=1
BIO_EMAIL=your.email@example.com
```

## Key Implementation Details

### Streaming Analysis
The `/analyze` endpoint streams server-sent event chunks so the frontend can render progress incrementally and remain responsive.

### Conversation Memory
Thread metadata and user memory are stored in PostgreSQL, while full conversation turns and analysis history are persisted in MongoDB.

### RAG System
The backend ingests HuggingFace MedRAG corpora and PDF content into ChromaDB, then retrieves evidence-based documents to support psychiatric reasoning.

### GPU Optimization
The ingestion and embedding pipelines support CUDA acceleration when available.

## Notes

- Docker Compose mounts the backend and frontend folders for live development.
- The backend Dockerfile starts the Flask app using `python run.py`.
- The frontend service is configured to call the backend at `http://localhost:5000`.
- `host.docker.internal` is used to connect containers to local Ollama services.

## Last Updated

June 2026
