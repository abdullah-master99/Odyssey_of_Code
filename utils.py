import os
import pdfplumber
import docx
import chromadb
import logging
import logging.handlers
import json
import hashlib
import warnings
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from langchain_core.messages import HumanMessage

# Configure logging to suppress specific PDFMiner warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

# Define base directory and directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = {
    'data': {
        'rfps': os.path.join(BASE_DIR, 'data', 'rfps'),
        'company_data': os.path.join(BASE_DIR, 'data', 'company_data'),
        'evaluation_results': os.path.join(BASE_DIR, 'data', 'evaluation_results'),
        'feedback': os.path.join(BASE_DIR, 'data', 'feedback'),
        'cache': os.path.join(BASE_DIR, 'data', 'cache')
    },
    'embeddings': {
        'rfp': os.path.join(BASE_DIR, 'embeddings', 'rfp_embeddings'),
        'company': os.path.join(BASE_DIR, 'embeddings', 'company_embeddings')
    },
    'logs': os.path.join(BASE_DIR, 'logs'),
    'templates': os.path.join(BASE_DIR, 'templates')
}

# Ensure all directories exist
for category, path in DIRS.items():
    if isinstance(path, dict):
        for subpath in path.values():
            if isinstance(subpath, dict):
                for final_path in subpath.values():
                    os.makedirs(final_path, exist_ok=True)
            else:
                os.makedirs(subpath, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rfp_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize embedding model
try:
    embedding_model = SentenceTransformer('all-mpnet-base-v2')  # This model produces 768-dimensional embeddings
    logger.info(f"Loaded embedding model: all-mpnet-base-v2")
except Exception as e:
    logger.error(f"Error loading embedding model: {str(e)}")
    raise

# Initialize LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=2048
    )
    logger.info(f"Initialized LLM: {LLM_MODEL}")
except Exception as e:
    logger.error(f"Error initializing LLM: {str(e)}")
    raise

# Initialize ChromaDB clients for different collections
try:
    # Client for RFP embeddings
    rfp_chroma_client = chromadb.PersistentClient(path="embeddings/rfp_embeddings")
    rfp_collection = rfp_chroma_client.get_or_create_collection(
        name="rfp_documents",
        metadata={"description": "RFP document embeddings"}
    )
    
    # Client for company embeddings
    company_chroma_client = chromadb.PersistentClient(path="embeddings/company_embeddings")
    company_collection = company_chroma_client.get_or_create_collection(
        name="company_documents",
        metadata={"description": "Company document embeddings"}
    )
    
    logger.info("Initialized ChromaDB collections with separate persistent storage")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}")
    raise

def parse_pdf(file_path: str) -> Optional[str]:
    """Extract text from a PDF file"""
    try:
        # Suppress PDFMiner warnings about CropBox
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
            text_content = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            return "\n\n".join(text_content) if text_content else None
    
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {str(e)}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # Adjust chunk end to the nearest sentence or paragraph break
        if end < text_len:
            # Look for paragraph break
            next_para = text.find('\n\n', end - overlap, end + overlap)
            if next_para != -1:
                end = next_para
            else:
                # Look for sentence break
                next_sentence = text.find('. ', end - overlap, end + overlap)
                if next_sentence != -1:
                    end = next_sentence + 1
        else:
            end = text_len
            
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap if end < text_len else text_len
        
    return chunks

def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for a text using the configured model"""
    try:
        if not text:
            return None
        
        # Generate embedding
        embedding = embedding_model.encode(text)
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

def get_llm_response(prompt: str) -> str:
    """Get a response from the LLM"""
    try:
        # Format prompt as a chat message
        message = HumanMessage(content=prompt)
        
        # Get response from LLM
        response = llm.invoke([message])
        
        return response.content if response else "Sorry, I couldn't generate a response."
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return f"Error: {str(e)}"

def reset_collections():
    """Reset ChromaDB collections to handle embedding dimension changes"""
    try:
        # Reset RFP collection
        rfp_chroma_client.delete_collection("rfp_documents")
        global rfp_collection
        rfp_collection = rfp_chroma_client.create_collection(
            name="rfp_documents",
            metadata={"description": "RFP document embeddings"}
        )
        
        # Reset company collection
        company_chroma_client.delete_collection("company_documents")
        global company_collection
        company_collection = company_chroma_client.create_collection(
            name="company_documents",
            metadata={"description": "Company document embeddings"}
        )
        
        logger.info("Successfully reset ChromaDB collections")
    except Exception as e:
        logger.error(f"Error resetting collections: {str(e)}")
        raise

class ResultTracker:
    """Track and store evaluation results"""
    def __init__(self):
        self.results_dir = "data/evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_result(self, result: dict) -> str:
        """Save evaluation result and return the result ID"""
        result_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        result['timestamp'] = datetime.now().isoformat()
        
        filepath = os.path.join(self.results_dir, f"{result_id}.json")
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result_id

result_tracker = ResultTracker()

class FeedbackAnalyzer:
    """Analyze and store feedback for RFP evaluations"""
    def __init__(self):
        self.feedback_dir = os.path.join("data", "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def save_feedback(self, evaluation_id: str, feedback: str) -> bool:
        """Save feedback for an evaluation"""
        try:
            feedback_data = {
                "evaluation_id": evaluation_id,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            filepath = os.path.join(self.feedback_dir, f"{evaluation_id}_feedback.json")
            with open(filepath, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Feedback saved successfully for evaluation {evaluation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return False

feedback_analyzer = FeedbackAnalyzer()