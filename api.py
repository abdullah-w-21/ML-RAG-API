# Teaching Assistant API - Professional ML APIs Backend
# Tailored for API Fundamentals and FastAPI Education

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import PyPDF2
import pickle
import re
import os
import joblib
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="🎓 API Teaching Assistant",
    description="Professional ML APIs Backend with Teaching Assistant for API Fundamentals & FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware - Allow all origins for educational purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# RAG Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
PDF_FILE_PATH = "api_fundamentals_guide.pdf"  # Updated name for teaching content
VECTOR_STORE_PATH = "teaching_vectorstore.pkl"
CHUNK_SIZE = 1000  # Larger chunks for educational content
CHUNK_OVERLAP = 200

# =============================================================================
# HOUSE PRICE PREDICTION SETUP
# =============================================================================

# Load house price model
house_model = None
feature_names = None

try:
    house_model = joblib.load('house_price_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✅ House price model loaded successfully!")
except FileNotFoundError:
    print("❌ House price model files not found.")

# =============================================================================
# TEACHING ASSISTANT SETUP
# =============================================================================

# Configure Gemini AI
gemini_model = None
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("✅ Teaching Assistant AI configured successfully!")
    except Exception as e:
        print(f"❌ Teaching Assistant AI configuration failed: {e}")
else:
    print("⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable.")

# Knowledge base class for teaching content
class TeachingKnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.chunk_vectors = None
        self.metadata = {}
        self.loaded = False

knowledge_base = TeachingKnowledgeBase()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

# House price prediction models
class HousePredictionRequest(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft: int
    age: int
    location_score: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft": 1800,
                "age": 10,
                "location_score": 8
            }
        }

# Teaching assistant models
class TeachingQuestion(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is FastAPI and why should I use it for building APIs?"
            }
        }

# =============================================================================
# TEACHING ASSISTANT HELPER FUNCTIONS
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Failed to read teaching content PDF: {str(e)}")

def clean_text(text: str) -> str:
    """Clean and normalize text for better learning content processing"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks optimized for educational content"""
    text = clean_text(text)
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            # Look for natural break points (paragraphs, sections)
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 100:  # Minimum chunk size for educational content
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def create_vector_store(chunks: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Create TF-IDF vector store optimized for educational content"""
    vectorizer = TfidfVectorizer(
        max_features=8000,  # More features for educational content
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams for technical terms
        min_df=1,
        max_df=0.7  # Lower threshold for educational terms
    )
    
    chunk_vectors = vectorizer.fit_transform(chunks)
    return vectorizer, chunk_vectors

def find_relevant_chunks(question: str, top_k: int = 4) -> List[Dict]:
    """Find most relevant educational content chunks"""
    if not knowledge_base.loaded:
        return []
    
    try:
        question_vector = knowledge_base.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, knowledge_base.chunk_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for educational content
                relevant_chunks.append({
                    "text": knowledge_base.chunks[idx],
                    "similarity": float(similarities[idx]),
                    "chunk_id": int(idx)
                })
        
        return relevant_chunks
    
    except Exception as e:
        print(f"Error in educational content search: {e}")
        return []

def save_vector_store():
    """Save the teaching vector store to disk"""
    vector_store_data = {
        "chunks": knowledge_base.chunks,
        "vectorizer": knowledge_base.vectorizer,
        "chunk_vectors": knowledge_base.chunk_vectors,
        "metadata": knowledge_base.metadata
    }
    
    with open(VECTOR_STORE_PATH, 'wb') as f:
        pickle.dump(vector_store_data, f)
    
    print(f"✅ Teaching vector store saved to {VECTOR_STORE_PATH}")

def load_vector_store() -> bool:
    """Load existing teaching vector store from disk"""
    if not os.path.exists(VECTOR_STORE_PATH):
        return False
    
    try:
        with open(VECTOR_STORE_PATH, 'rb') as f:
            vector_store_data = pickle.load(f)
        
        knowledge_base.chunks = vector_store_data["chunks"]
        knowledge_base.vectorizer = vector_store_data["vectorizer"]
        knowledge_base.chunk_vectors = vector_store_data["chunk_vectors"]
        knowledge_base.metadata = vector_store_data["metadata"]
        knowledge_base.loaded = True
        
        print(f"✅ Teaching vector store loaded from {VECTOR_STORE_PATH}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to load teaching vector store: {e}")
        return False

def initialize_teaching_knowledge_base():
    """Initialize teaching knowledge base from educational PDF"""
    print("🎓 Initializing Teaching Assistant knowledge base...")
    
    # Try to load existing vector store first
    if load_vector_store():
        print(f"📚 Teaching knowledge base loaded with {len(knowledge_base.chunks)} educational chunks")
        return
    
    # Check if educational PDF file exists
    if not os.path.exists(PDF_FILE_PATH):
        print(f"⚠️ Educational PDF not found: {PDF_FILE_PATH}")
        print("Please ensure you have the API fundamentals teaching guide PDF in place.")
        return
    
    try:
        print(f"📖 Processing educational content: {PDF_FILE_PATH}")
        
        # Extract text from educational PDF
        text = extract_text_from_pdf(PDF_FILE_PATH)
        
        if not text.strip():
            raise Exception("No educational content found in PDF")
        
        # Create educational chunks
        chunks = chunk_text(text)
        print(f"📄 Created {len(chunks)} educational content chunks")
        
        # Create vector store for educational content
        vectorizer, chunk_vectors = create_vector_store(chunks)
        
        # Store in teaching knowledge base
        knowledge_base.chunks = chunks
        knowledge_base.vectorizer = vectorizer
        knowledge_base.chunk_vectors = chunk_vectors
        knowledge_base.metadata = {
            "source_file": PDF_FILE_PATH,
            "content_type": "API Fundamentals & FastAPI Teaching Guide",
            "total_chunks": len(chunks),
            "total_characters": len(text),
            "vector_features": chunk_vectors.shape[1],
            "subject": "API Development Education"
        }
        knowledge_base.loaded = True
        
        # Save vector store for future use
        save_vector_store()
        
        print(f"✅ Teaching Assistant knowledge base initialized successfully!")
        print(f"   - {len(chunks)} educational chunks processed")
        print(f"   - {chunk_vectors.shape[1]} vector features for teaching content")
        
    except Exception as e:
        print(f"❌ Failed to initialize teaching knowledge base: {e}")

# Initialize teaching knowledge base at startup
@app.on_event("startup")
async def startup_event():
    initialize_teaching_knowledge_base()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Welcome endpoint for the Teaching Assistant API"""
    return {
        "message": "🎓 API Teaching Assistant",
        "description": "Professional ML APIs Backend with Educational Support",
        "purpose": "Teaching API Fundamentals & FastAPI Development",
        "services": {
            "house_prediction": {
                "status": "ready" if house_model is not None else "not_ready",
                "endpoint": "/predict",
                "description": "Practical ML API demonstration"
            },
            "teaching_assistant": {
                "status": "ready" if knowledge_base.loaded and gemini_model else "not_ready",
                "endpoint": "/ask",
                "description": "AI teaching assistant for API and FastAPI questions"
            }
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for teaching environment"""
    return {
        "status": "healthy",
        "environment": "teaching",
        "services": {
            "house_model": house_model is not None,
            "teaching_assistant": gemini_model is not None,
            "knowledge_base": knowledge_base.loaded,
            "educational_chunks": len(knowledge_base.chunks) if knowledge_base.loaded else 0
        },
        "ready_for_teaching": all([
            house_model is not None,
            gemini_model is not None,
            knowledge_base.loaded
        ])
    }

# =============================================================================
# HOUSE PRICE PREDICTION ENDPOINTS (Teaching Demo)
# =============================================================================

@app.post("/predict")
async def predict_house_price(request: HousePredictionRequest):
    """
    Predict house price - Practical ML API demonstration for students
    
    This endpoint serves as a real-world example of how ML models are deployed in production APIs.
    Students can use this to understand request/response patterns, data validation, and error handling.
    """
    if house_model is None:
        raise HTTPException(
            status_code=503, 
            detail="ML model not available. This demonstrates proper error handling in production APIs."
        )
    
    try:
        # Convert request to model input format
        features = np.array([[
            request.bedrooms,
            request.bathrooms,
            request.sqft,
            request.age,
            request.location_score
        ]])
        
        # Make prediction
        prediction = house_model.predict(features)[0]
        
        # Calculate additional business metrics
        price_per_sqft = prediction / request.sqft
        
        # Business logic for categorization
        if prediction < 200000:
            price_category = "Budget-friendly"
        elif prediction < 400000:
            price_category = "Mid-range"
        elif prediction < 600000:
            price_category = "Premium"
        else:
            price_category = "Luxury"
        
        return {
            "success": True,
            "service": "house_prediction_demo",
            "educational_note": "This demonstrates a complete ML API response with business logic",
            "input_features": {
                "bedrooms": request.bedrooms,
                "bathrooms": request.bathrooms,
                "sqft": request.sqft,
                "age": request.age,
                "location_score": request.location_score
            },
            "prediction": {
                "price": round(prediction, 2),
                "formatted_price": f"${prediction:,.0f}",
                "price_per_sqft": round(price_per_sqft, 2),
                "category": price_category
            },
            "model_info": {
                "type": "Linear Regression",
                "features_used": len(feature_names) if feature_names else 5,
                "purpose": "Educational demonstration of ML API patterns"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}. This demonstrates proper error handling."
        )

@app.get("/model-info")
async def get_house_model_info():
    """Get information about the demo ML model - Educational endpoint"""
    if house_model is None:
        raise HTTPException(status_code=503, detail="Demo ML model not loaded")
    
    return {
        "service": "educational_ml_demo",
        "model_type": "Linear Regression",
        "purpose": "Teaching ML API development patterns",
        "features": feature_names if feature_names else [
            "bedrooms", "bathrooms", "sqft", "age", "location_score"
        ],
        "feature_descriptions": {
            "bedrooms": "Number of bedrooms (1-5)",
            "bathrooms": "Number of bathrooms (1-3)", 
            "sqft": "Square footage (800-3500)",
            "age": "Age of house in years (0-50)",
            "location_score": "Location quality score (1-10, higher is better)"
        },
        "target": "House price in USD",
        "training_info": "Trained on synthetic data for educational purposes",
        "learning_objectives": [
            "Understand ML model deployment",
            "Learn API request/response patterns",
            "Practice data validation",
            "Explore error handling strategies"
        ]
    }

# =============================================================================
# TEACHING ASSISTANT ENDPOINTS
# =============================================================================

@app.post("/ask")
async def ask_teaching_assistant(request: TeachingQuestion):
    """
    Ask the Teaching Assistant about API fundamentals and FastAPI
    
    This AI assistant has access to comprehensive educational content about API development,
    FastAPI framework, best practices, and practical examples.
    """
    if gemini_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Teaching Assistant AI not configured. Please set GEMINI_API_KEY environment variable."
        )
    
    if not knowledge_base.loaded:
        raise HTTPException(
            status_code=503,
            detail="Teaching knowledge base not loaded. Educational content is not available."
        )
    
    try:
        # Find relevant educational content
        relevant_chunks = find_relevant_chunks(request.question, top_k=4)
        
        if not relevant_chunks:
            # Provide general guidance when no specific content is found
            prompt = f"""
You are a professional Teaching Assistant specializing in API development and FastAPI.

A student asked: "{request.question}"

While I don't have specific content from our teaching materials that directly addresses this question, 
I can provide general guidance based on best practices in API development and FastAPI.

Please provide a helpful, educational response that:
- Answers the student's question clearly and concisely
- Relates to API fundamentals and FastAPI where possible
- Encourages further learning
- Suggests they refer to the official documentation for detailed information
- Maintains a supportive, professional teaching tone

Response:"""
            
            response = gemini_model.generate_content(prompt)
            
            return {
                "success": True,
                "service": "teaching_assistant",
                "question": request.question,
                "answer": response.text,
                "context_used": False,
                "relevant_sections": 0,
                "teaching_note": "General guidance provided - refer to course materials for detailed information"
            }
        
        # Create educational context from relevant content
        context_parts = []
        sources = []
        
        for i, chunk_data in enumerate(relevant_chunks):
            context_parts.append(f"Teaching Material {i+1}:\n{chunk_data['text']}")
            sources.append(f"Section {chunk_data['chunk_id']} (relevance: {chunk_data['similarity']:.2f})")
        
        context = "\n\n".join(context_parts)
        
        # Create specialized teaching prompt
        prompt = f"""
You are a professional Teaching Assistant specializing in API development and FastAPI. You are helping students learn during a hands-on coding session about API fundamentals.

Here is relevant content from our teaching materials:

{context}

Student Question: "{request.question}"

Please provide a comprehensive, educational response that:

1. **Directly answers the student's question** using the provided teaching materials
2. **Explains concepts clearly** with practical examples where appropriate
3. **Relates to the hands-on session** they're participating in
4. **Encourages deeper understanding** by explaining the "why" behind concepts
5. **Provides actionable guidance** they can apply immediately
6. **Uses a supportive, professional teaching tone** appropriate for university students
7. **References specific parts** of the teaching materials when relevant
8. **Suggests practical next steps** for further learning

If the question relates to the demo house price prediction API, use that as a practical example to illustrate concepts.

Format your response to be clear, well-structured, and educational. Use bullet points or numbered lists when it helps clarify complex concepts.

Teaching Assistant Response:"""
        
        # Generate educational response
        response = gemini_model.generate_content(prompt)
        
        return {
            "success": True,
            "service": "teaching_assistant",
            "question": request.question,
            "answer": response.text,
            "context_used": True,
            "relevant_sections": len(relevant_chunks),
            "sources": sources,
            "teaching_materials": knowledge_base.metadata.get("source_file", "API Fundamentals Guide"),
            "session_topic": "API Fundamentals & FastAPI Development"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Teaching Assistant error: {str(e)}")

@app.post("/search-materials")
async def search_teaching_materials(request: TeachingQuestion):
    """
    Search through teaching materials without AI interpretation
    
    Useful for students who want to find specific sections in the educational content.
    """
    if not knowledge_base.loaded:
        raise HTTPException(
            status_code=503,
            detail="Teaching materials not loaded."
        )
    
    try:
        relevant_chunks = find_relevant_chunks(request.question, top_k=6)
        
        return {
            "success": True,
            "service": "materials_search",
            "query": request.question,
            "results": [
                {
                    "section_id": chunk["chunk_id"],
                    "preview": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                    "full_content": chunk["text"],
                    "relevance_score": chunk["similarity"]
                }
                for chunk in relevant_chunks
            ],
            "total_results": len(relevant_chunks),
            "search_tip": "Use specific technical terms for better results"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Materials search failed: {str(e)}")

@app.get("/teaching-info")
async def get_teaching_info():
    """Get information about the teaching assistant and available materials"""
    if not knowledge_base.loaded:
        return {
            "loaded": False,
            "service": "teaching_assistant",
            "message": "Teaching materials not loaded. Please ensure educational PDF exists and restart API."
        }
    
    # Get sample educational content
    sample_sections = []
    for i, chunk in enumerate(knowledge_base.chunks[:3]):
        sample_sections.append({
            "section_id": i,
            "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
        })
    
    return {
        "loaded": True,
        "service": "teaching_assistant",
        "subject": "API Fundamentals & FastAPI Development",
        "metadata": knowledge_base.metadata,
        "total_sections": len(knowledge_base.chunks),
        "search_capabilities": f"{knowledge_base.chunk_vectors.shape[1]} dimensional vector search",
        "sample_content": sample_sections,
        "teaching_features": [
            "Context-aware responses",
            "Educational material integration",
            "Practical examples with demo API",
            "Progressive difficulty explanations",
            "Professional teaching tone"
        ],
        "session_focus": "Hands-on API development with FastAPI"
    }

# =============================================================================
# EDUCATIONAL UTILITIES
# =============================================================================

@app.get("/demo-endpoints")
async def list_demo_endpoints():
    """List all available endpoints for educational exploration"""
    return {
        "message": "Available API endpoints for learning",
        "house_prediction_demo": {
            "POST /predict": "Predict house prices (ML API demo)",
            "GET /model-info": "Get ML model information"
        },
        "teaching_assistant": {
            "POST /ask": "Ask questions about APIs and FastAPI",
            "POST /search-materials": "Search through teaching materials",
            "GET /teaching-info": "Get teaching assistant information"
        },
        "system": {
            "GET /": "API information and status",
            "GET /health": "System health check",
            "GET /demo-endpoints": "This endpoint list",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "learning_tip": "Try the /docs endpoint to explore the interactive API documentation!"
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    print("🎓 Starting API Teaching Assistant...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🏠 House Price Demo: http://localhost:8000/predict")
    print("🤖 Teaching Assistant: http://localhost:8000/ask")
    print("📚 Teaching Materials: http://localhost:8000/teaching-info")
    uvicorn.run(app, host="0.0.0.0", port=8000)
