import uuid
import json
import logging
from typing import Optional
from fastapi import FastAPI, Request, Response, Cookie, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import your custom modules
from .analyzer import SymptomAnalyzer
from .rag_ingestion import MedicalRAGIngestion
from .memory import ConversationMemory

# Setup
logger = logging.getLogger(__name__)
symptom_analyzer = SymptomAnalyzer()
rag_ingestion = MedicalRAGIngestion()

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-ID"] # Essential for your frontend to see this header
)

@app.get("/me")
async def get_user_info(response: Response, user_id: Optional[str] = Cookie(None)):
    if not user_id:
        user_id = str(uuid.uuid4())
        logger.info("No user_id cookie found. Generating new user_id.")
        # Set cookie directly on the response object
        response.set_cookie(key="user_id", value=user_id, max_age=60*60*24*365)
    return {"user_id": user_id}

@app.get("/threads")
async def get_user_threads(user_id: Optional[str] = Cookie(None)):
    if not user_id:
        raise HTTPException(status_code=401, detail="User not identified")
    
    memory = ConversationMemory(symptom_analyzer.db_manager, user_id)
    threads = memory.fetch_user_threads()
    logger.info(f"User {user_id} requested threads. Found {len(threads)} threads.")
    return threads

@app.get("/thread/{thread_id}")
async def get_thread(thread_id: str, user_id: Optional[str] = Cookie(None)):
    if not user_id:
        raise HTTPException(status_code=404, detail="User not found")
    
    memory = ConversationMemory(symptom_analyzer.db_manager, user_id, thread_id)
    logger.info(f"User {user_id} requested thread {thread_id}.")
    return memory.fetch_thread_history()

@app.post("/analyze")
async def analyze_symptoms(request: Request, user_id: Optional[str] = Cookie(None)):
    data = await request.json()
    user_query = data.get('query', '')
    thread_id = data.get('thread_id', str(uuid.uuid4()))

    async def event_generator():
       
        for event in symptom_analyzer.analyze(user_query, user_id, thread_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={"X-Thread-ID": thread_id}
    )

@app.post("/fetch-analysis")
async def fetch_analysis(request: Request, user_id: Optional[str] = Cookie(None)):
    data = await request.json()
    thread_id = data.get('thread_id')
    
    if not user_id or not thread_id:
        raise HTTPException(status_code=400, detail="Invalid request")
        
    memory = ConversationMemory(symptom_analyzer.db_manager, user_id, thread_id)
    return memory.fetch_final_analysis()

@app.get("/reset-rag")
async def reset_rag():
    rag_ingestion.ingest_medrag_corpus()
    return {"message": "RAG pipeline reset successfully!"}
