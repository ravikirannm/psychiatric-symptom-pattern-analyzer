# Evidence-Based Symptom Analyzer

## Overview

**Evidence-Based Symptom Analyzer** is an intelligent medical consultation application that leverages artificial intelligence and evidence-based medical data to analyze patient symptoms and provide clinically relevant insights. The application combines natural language processing, retrieval-augmented generation (RAG), and conversational AI to deliver structured clinical analysis of medical presentations.

![Application Screenshot](image.png)

## Key Features

- **Symptom Analysis**: Converts natural language symptom descriptions into structured clinical presentations
- **Conversational Memory**: Maintains conversation history and context across multiple threads for personalized analysis
- **Medical RAG Integration**: Retrieves evidence-based medical information from a curated corpus to support analysis
- **Clinical Reasoning**: Employs multi-pass clinical reasoning to ensure comprehensive symptom coverage
- **User Sessions**: Manages individual user sessions with persistent conversation threads

## Application Architecture

### Frontend
- **Framework**: Angular 21.2.0 (latest standalone components)
- **Styling**: SCSS with Bootstrap 5.3.3
- **UI Features**:
  - Interactive symptom input interface
  - Conversation thread management
  - Markdown rendering of medical analysis
  - Security-conscious HTML sanitization

### Backend
- **Framework**: Flask with CORS support
- **Port**: 5000
- **Core Capabilities**:
  - RESTful API endpoints for symptom analysis
  - User session and cookie-based authentication
  - Conversation thread management
  - Real-time streaming responses

### AI & ML Stack
- **LLM**: Ollama with Qwen3.5:9b model for clinical reasoning
- **NLP**: 
  - Transformers library for semantic analysis
  - PyTorch for deep learning operations
  - GPU acceleration support (CUDA compatible)
- **RAG System**:
  - ChromaDB for vector database and embeddings
  - Medical corpus ingestion pipeline
  - Semantic retrieval of medical information
- **Medical Data Processing**:
  - PyMuPDF for document processing
  - BioPython for biological sequence analysis
  - HuggingFace datasets integration

### Database & Storage
- **Primary**: MongoDB (document-based medical records and conversation history)
- **Secondary**: PostgreSQL (relational data with psycopg2)
- **Vector DB**: ChromaDB (semantic embeddings for medical corpus)

## Technology Stack

### Backend Dependencies
```
flask              - Web framework
pymongo            - MongoDB driver
psycopg2-binary    - PostgreSQL driver
requests           - HTTP client
ollama             - LLM interaction
transformers       - NLP models
torch              - Deep learning framework
torchvision        - Computer vision models
python-dotenv      - Environment configuration
chromadb           - Vector database
pymupdf            - PDF processing
datasets           - HuggingFace datasets
biopython          - Biological data processing
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

1. **User Input**: Patient describes symptoms in natural language
2. **Session Management**: Application identifies or creates user session with unique ID
3. **Clinical Reformulation**: LLM converts symptom description into 5 structured clinical variations
4. **Medical Retrieval**: RAG system retrieves relevant evidence-based medical information
5. **Query Verification**: Validates clinical relevance of extracted medical information
6. **Analysis Output**: Generates structured clinical analysis with supporting evidence
7. **Memory Persistence**: Stores conversation history for context in future interactions

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
Ollama LLM (Qwen3.5:9b)
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
├── backend/                    # Python Flask backend
│   ├── engine/
│   │   ├── analyzer.py        # Main symptom analysis engine
│   │   ├── main.py            # Flask app and routes
│   │   ├── rag_ingestion.py   # Medical corpus ingestion
│   │   ├── rag_retriever.py   # RAG retrieval system
│   │   ├── verifier.py        # Query verification
│   │   ├── memory.py          # Conversation memory management
│   │   ├── database.py        # Database operations
│   │   ├── data_model.py      # Data structures
│   │   └── query_preprocess.py # Query preprocessing
│   ├── medcpt_db/             # ChromaDB vector store
│   ├── run.py                 # Entry point
│   ├── constants.py           # Configuration constants
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
│
├── frontend/                   # Angular application
│   ├── src/
│   │   ├── app/
│   │   │   ├── main/          # Main component
│   │   │   ├── services/
│   │   │   │   └── api.service.ts  # Backend API client
│   │   │   ├── app.routes.ts  # Application routing
│   │   │   └── interfaces.ts  # TypeScript interfaces
│   │   ├── main.ts            # Bootstrap file
│   │   ├── index.html         # Root HTML
│   │   └── styles.scss        # Global styles
│   ├── angular.json           # Angular CLI configuration
│   ├── package.json           # NPM dependencies
│   └── tsconfig.json          # TypeScript configuration
│
├── docker-compose.yml         # Container orchestration
└── README.md                  # This file
```

## API Endpoints

### User Management
- `GET /me` - Get or create user session with unique ID
- `GET /threads` - Retrieve all conversation threads for the user
- `GET /thread/<thread_id>` - Get conversation history for a specific thread

### Analysis
- `POST /analyze` - Analyze symptoms (streaming response)

## Running the Application

### Using Docker Compose
```bash
docker-compose up
```
This will start both the backend (port 5000) and frontend (port 4200).

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

The application will be accessible at `http://localhost:4200`

## Configuration

Create a `.env` file in the backend directory with the following variables:
```
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b
MONGODB_URI=mongodb://localhost:27017/symptom_analyzer
POSTGRES_URI=postgresql://user:password@localhost:5432/symptom_analyzer
```

## Key Implementation Details

### Streaming Analysis
The `/analyze` endpoint uses Flask's streaming capabilities to provide real-time progress updates as the LLM analyzes symptoms, creating a responsive user experience.

### Conversation Memory
The system maintains multi-level context:
- **Thread Memory**: Conversation-specific context
- **Shared Memory**: User-wide persistent information
- **Session History**: Full conversation logs

### RAG System
Medical documents are ingested into ChromaDB with semantic embeddings, enabling intelligent retrieval of relevant medical information to support clinical analysis.

### GPU Optimization
The system supports CUDA-enabled GPUs for accelerated deep learning operations, automatically detected at startup.

## Team & Attribution

Built with AI-assisted development using OpenAI's GPT and GitHub Copilot, integrating best practices in medical informatics and evidence-based medicine.

---

**Last Updated**: May 2026
