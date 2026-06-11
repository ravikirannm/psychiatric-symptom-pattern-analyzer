from dotenv import load_dotenv
load_dotenv()

import logging
import torch
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.info("Starting the Symptom Analyzer FastAPI backend...")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

# Start the application using Uvicorn
if __name__ == '__main__':
    # 'main_fast:app' assumes your file is named main_fast.py 
    # and the FastAPI instance is named 'app'
    uvicorn.run(
        "main_fast:app", 
        host='0.0.0.0', 
        port=5000, 
        reload=True,  # This replaces Flask's debug=True
        log_level="info"
    )
